import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import os
import random
import copy
import sys
import yaml
from metrics import ClassificationMetrics
from utils import LabelSmoothingCrossEntropyLoss
from logger import Logger
from model import ResNet50
from load_dataset import load_ham10000, get_transforms, SkinData, dataset_iid, federated_dataset_split, load_odir, ODIRDataFrame
import time

# Load parameters
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

num_classes = 8

# Complete model training definition
class complete_model:
    def __init__(self, idx, net, lr, device, dataset_train, dataset_test_onsite, global_model):
        """
        Args:
            idx: Client ID for federated learning
            net: Initialized model
            lr: Learning rate
            device: Computation device
            dataset_train: Training dataset
            dataset_test_onsite: Test dataset
            global_model: Global model for FedProx regularization
        """
        self.idx = idx
        self.net = copy.deepcopy(net).to(device)
        self.device = device

        # Loss functions
        self.criterion = LabelSmoothingCrossEntropyLoss(smoothing=0.1)
        self.criterion_eval = nn.CrossEntropyLoss()

        # Data loaders
        self.train_loader = DataLoader(dataset_train, batch_size=config['batch_size'], shuffle=True)
        self.test_loader = DataLoader(dataset_test_onsite, batch_size=config['batch_size'], shuffle=False)

        # Evaluation metrics
        self.metrics = ClassificationMetrics(num_classes=num_classes)

        # Optimizer and learning rate warmup
        self.warmup_epochs = 0
        self.current_epoch = 0
        self.lr = lr
        self.optimizer = torch.optim.Adam(
            self.net.parameters(),
            lr=float(config['lr']),
            weight_decay=float(config['weight_decay'])
        )

        # Learning rate scheduler
        self.scheduler = self._get_scheduler(lr)

        # FedProx regularization parameter
        self.lambda_prox = config['lambda_prox']
        self.global_model = global_model

    def train(self):
        """Local training"""
        self.net.train()
        self.metrics.reset()
        batch_loss = []

        print(f"Training set size: {len(self.train_loader.dataset)}")

        # Learning rate warmup - linear increase
        if self.current_epoch < self.warmup_epochs:
            warmup_factor = (self.current_epoch + 1) / self.warmup_epochs
            lr = self.lr * warmup_factor
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

        self.current_epoch += 1

        # Save a copy of the global model parameters for FedProx
        global_weights = copy.deepcopy(list(self.global_model.parameters()))

        for batch in self.train_loader:
            images = batch['image'].to(self.device)
            labels = batch['labels'].to(self.device)
            self.optimizer.zero_grad()
            outputs = self.net(images)

            # Standard loss
            loss = self.criterion(outputs, labels)

            # FedProx: proximal term
            prox_loss = 0.0
            for p_i, param in enumerate(self.net.parameters()):
                prox_loss += (self.lambda_prox / 2) * torch.norm((param - global_weights[p_i])) ** 2

            total_loss = loss + prox_loss

            total_loss.backward()
            self.optimizer.step()

            batch_loss.append(loss.item())
            self.metrics.update(outputs, labels)

        # Compute metrics
        results = self.metrics.compute()
        return (
            np.mean(batch_loss),
            self.net.state_dict(),
            results['kappa'],
            results['f1'],
            results['auc'],
            results['final_score'],
            results['accuracy'],
            results['precision'],
            results['recall'],
        )

    def evaluate(self):
        """Local evaluation"""
        self.net.eval()
        self.metrics.reset()
        test_loader = self.test_loader

        with torch.no_grad():
            batch_loss = []
            for batch in test_loader:
                images = batch['image'].to(self.device)
                labels = batch['labels'].to(self.device)
                outputs = self.net(images)
                self.metrics.update(outputs, labels)
                loss = self.criterion_eval(outputs, labels)
                batch_loss.append(loss.item())

        # Compute metrics
        results = self.metrics.compute()
        return (
            np.mean(batch_loss),
            results['kappa'],
            results['f1'],
            results['auc'],
            results['final_score'],
            results['accuracy'],
            results['precision'],
            results['recall'],
        )

    def _get_scheduler(self, lr):
        """Return the corresponding learning rate scheduler based on config"""
        if config['lr_scheduler'] == 'StepLR':
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=config['lr_step_size'],
                gamma=config['lr_gamma']
            )
        elif config['lr_scheduler'] == 'ReduceLROnPlateau':
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=config['lr_gamma'],
                patience=config['lr_patience'],
                min_lr=config['lr_min']
            )
        elif config['lr_scheduler'] == 'CosineAnnealingLR':
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=config['lr_T_max'],
                eta_min=config['lr_min']
            )
        else:
            return None

    def update_scheduler(self, metric=None):
        """Update learning rate scheduler"""
        if self.current_epoch <= self.warmup_epochs:
            return

        if self.scheduler is None:
            return

        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            if metric is not None:
                self.scheduler.step(metric)
        else:
            self.scheduler.step()

        current_lr = self.optimizer.param_groups[0]['lr']
        print(f"current learning rate: {current_lr:.2e}")


# FedAvg algorithm
def federated_avg(w_locals, dataset_sizes):
    """Federated averaging algorithm (weighted version)"""
    w_avg = copy.deepcopy(w_locals[0])
    total_data_size = sum(dataset_sizes)
    for key in w_avg.keys():
        weighted_sum = torch.zeros_like(w_avg[key], dtype=torch.float32)
        for i in range(len(w_locals)):
            weight = (dataset_sizes[i] / total_data_size)
            weight_tensor = weight * torch.ones_like(w_locals[i][key], dtype=torch.float32, device=w_locals[i][key].device)
            weighted_sum += weight_tensor * w_locals[i][key].float()
        w_avg[key] = weighted_sum
    return w_avg

# ============================= Main Training Process ==============================
def main():

    # Prepare dataset
    print("Preparing dataset...")
    # HAM10000 dataset
    train_df, test_df = load_ham10000(config['metadata_path'], config['ham10000_image_root'])
    train_transforms, test_transforms = get_transforms()
    train_datasets = SkinData(train_df, transform=train_transforms)
    test_datasets = SkinData(test_df, transform=test_transforms)

    # ODIR dataset (commented out)
    # train_df, test_df = load_odir(
    #     train_dir = config['train_image_root'],
    #     test_dir  = config['offsite_test_image_root']
    # )
    # train_transforms, test_transforms = get_transforms()
    # train_datasets = ODIRDataFrame(train_df, transform=train_transforms, return_multihot=False)
    # test_datasets = ODIRDataFrame(test_df, transform=test_transforms, return_multihot=False)

    client_datasets, client_dict = federated_dataset_split(
        train_datasets,
        num_clients=5,
        partition_mode="dirichlet",
        balance=False,
        dir_alpha=0.5,
        save_dir="ham_dirichlet_unbalanced",
        load_existing=True
    )

    # Model initialization
    print("Initializing model...")
    global_model = ResNet50(num_classes=num_classes, pretrained=True)
    global_model.to(device)

    # Metrics collection
    train_metrics = { 'epoch': [], 'loss': [], 'kappa': [], 'f1': [], 'auc': [], 'final': [], 'accuracy': [], 'precision': [], 'recall': []}
    test_metrics = { 'epoch':[], 'loss':[], 'kappa': [], 'f1': [], 'auc': [], 'final': [], 'accuracy': [], 'precision': [], 'recall': []}

    # Client initialization
    print("Initializing clients...")
    clients = []
    dataset_sizes = []
    for idx in range(config['num_clients']):
        client = complete_model(
                idx=idx,
                net=global_model,
                lr=float(config['lr']),
                device=device,
                dataset_train=client_datasets[idx],
                dataset_test_onsite=test_datasets,
                global_model=global_model
            )
        clients.append(client)
        dataset_sizes.append(len(train_datasets[idx]))

    # Training loop
    print("Starting model training...")
    for epoch in range(config['epochs']):
        start_time = time.time()
        w_locals = []

        # For each epoch, collect client metrics
        loss_train, kappa_train, f1_train, auc_train, final_train, accuracy_train, preciseion_train, recall_train = [], [], [], [], [], [], [], []
        loss_offsite_test, kappa_offsite_test, f1_offsite_test, auc_offsite_test, final_offsite_test, accuracy_offsite_train, precisieon_offsie_train, recall_offsite_train = [], [], [], [], [], [], [], []

        for idx, client in enumerate(clients):
            # train
            loss, w, kappa, f1, auc, final, accuracy, precision, recall = client.train()
            w_locals.append(w)
            loss_train.append(loss)
            kappa_train.append(kappa)
            f1_train.append(f1)
            auc_train.append(auc)
            final_train.append(final)
            accuracy_train.append(accuracy)
            preciseion_train.append(precision)
            recall_train.append(recall)

            # offsite validation
            loss, kappa, f1, auc, final, accuracy, precision, recall = client.evaluate()
            loss_offsite_test.append(loss)
            kappa_offsite_test.append(kappa)
            f1_offsite_test.append(f1)
            auc_offsite_test.append(auc)
            final_offsite_test.append(final)
            accuracy_offsite_train.append(accuracy)
            precisieon_offsie_train.append(precision)
            recall_offsite_train.append(recall)

        # Update learning rate scheduler
        for idx, client in enumerate(clients):
            client.update_scheduler()

        # FedProx
        if (epoch+1) % config['fedavg_num'] == 0:
            print("FedProx...")
            w_global = federated_avg(w_locals, dataset_sizes)
            for idx, client in enumerate(clients):
                client.net.load_state_dict(w_global)
                client.global_model = copy.deepcopy(client.net)

        # Update metrics
        train_metrics['epoch'].append(epoch+1)
        test_metrics['epoch'].append(epoch+1)

        # train
        train_metrics['loss'].append(np.mean(loss_train))
        train_metrics['kappa'].append(np.mean(kappa_train))
        train_metrics['f1'].append(np.mean(f1_train))
        train_metrics['auc'].append(np.mean(auc_train))
        train_metrics['final'].append(np.mean(final_train))
        train_metrics['accuracy'].append(np.mean(accuracy_train))
        train_metrics['precision'].append(np.mean(preciseion_train))
        train_metrics['recall'].append(np.mean(recall_train))
        # offsite test
        test_metrics['loss'].append(np.mean(loss_offsite_test))
        test_metrics['kappa'].append(np.mean(kappa_offsite_test))
        test_metrics['f1'].append(np.mean(f1_offsite_test))
        test_metrics['auc'].append(np.mean(auc_offsite_test))
        test_metrics['final'].append(np.mean(final_offsite_test))
        test_metrics['accuracy'].append(np.mean(accuracy_offsite_train))
        test_metrics['precision'].append(np.mean(precisieon_offsie_train))
        test_metrics['recall'].append(np.mean(recall_offsite_train))

        # Print progress
        print(f"Epoch {epoch+1}/{config['epochs']}")
        print(f"train dataset - Kappa: {train_metrics['kappa'][-1]:.4f} | F1: {train_metrics['f1'][-1]:.4f} | "
              f"AUC: {train_metrics['auc'][-1]:.4f} | final score: {train_metrics['final'][-1]:.4f} | accuracy: {train_metrics['accuracy'][-1]:.4f} | ")
        print(f"offsite test  - Kappa: {test_metrics['kappa'][-1]:.4f} | F1: {test_metrics['f1'][-1]:.4f} | "
              f"AUC: {test_metrics['auc'][-1]:.4f} | final score: {test_metrics['final'][-1]:.4f} | accuracy: {test_metrics['accuracy'][-1]:.4f} | ")

        # Training time per epoch
        end_time = time.time()
        print(f"Epoch {epoch+1} training time: {end_time - start_time:.2f} seconds")

    # ============== Save results ===================
    os.makedirs('results', exist_ok=True)
    file_name = config['resultfile']
    train_metrics_df = pd.DataFrame(train_metrics)
    test_metrics_df = pd.DataFrame(test_metrics)
    with pd.ExcelWriter(file_name) as writer:
        train_metrics_df.to_excel(writer, sheet_name='Train Metrics', index=False)
        test_metrics_df.to_excel(writer, sheet_name='offsite test Metrics', index=False)
    print(f"Training metrics saved to {file_name}")


if __name__ == "__main__":
    log_file = open(config['recordfile'], 'a')
    original_stdout = sys.stdout
    sys.stdout = Logger(log_file)

    print("All parameter configurations:")
    for key, value in config.items():
        print(f"{key}: {value}")

    main()

    sys.stdout = original_stdout
    log_file.close()