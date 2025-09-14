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
from load_dataset import federated_dataset_split, load_ham10000, get_transforms, SkinData, load_odir, ODIRDataFrame
import time

# Load parameters
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Number of classes for training data
num_classes = 8

# Complete model training definition
class complete_model:
    def __init__(self, idx, net, lr, device, dataset_train, dataset_test_onsite):
        """
        Args:
            idx: Client ID for federated learning
            net: Initialized model
            lr: Learning rate
            device: Computation device
            dataset_train: Training dataset
            dataset_test_onsite: Test dataset
        """
        self.idx = idx
        self.net = copy.deepcopy(net).to(device)
        self.device = device

        # Loss functions
        self.criterion = LabelSmoothingCrossEntropyLoss(smoothing=0.1, num_classes=num_classes)   # Label smoothing for training
        self.criterion_eval = nn.CrossEntropyLoss()  # CrossEntropyLoss for evaluation

        # Data loaders
        self.train_loader = DataLoader(dataset_train, batch_size=config['batch_size'], shuffle=True)
        self.test_loader = DataLoader(dataset_test_onsite, batch_size=config['batch_size'], shuffle=False)

        # Evaluation metrics
        self.metrics = ClassificationMetrics(num_classes=num_classes)

        # Optimizer and learning rate warmup
        self.warmup_epochs = 0  # Number of warmup epochs
        self.current_epoch = 0  # Track current epoch
        self.lr = lr            # Initial learning rate (warmup target)
        self.optimizer = torch.optim.Adam(
            self.net.parameters(),
            lr=float(config['lr']),
            weight_decay=float(config['weight_decay'])
        )

        # Learning rate scheduler
        self.scheduler = self._get_scheduler(lr)

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

        self.current_epoch += 1  # Update epoch count

        for batch in self.train_loader:
            images = batch['image'].to(self.device)
            labels = batch['labels'].to(self.device)

            self.optimizer.zero_grad()
            outputs = self.net(images)
            loss = self.criterion(outputs, labels)

            loss.backward()
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
        # Only start scheduler after warmup
        if self.current_epoch <= self.warmup_epochs:
            return

        if self.scheduler is None:
            return

        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            if metric is not None:
                self.scheduler.step(metric)
        else:
            self.scheduler.step()

        # Print current learning rate
        current_lr = self.optimizer.param_groups[0]['lr']
        print(f"current learning rate: {current_lr:.2e}")

# FedAvg algorithm - weighted average
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

def evaluate_global_model(model, test_loader, device):
    model.eval()
    metrics = ClassificationMetrics(num_classes=num_classes)
    criterion = nn.CrossEntropyLoss()
    batch_loss = []
    with torch.no_grad():
        for batch in test_loader:
            images = batch['image'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            batch_loss.append(loss.item())
            metrics.update(outputs, labels)
    results = metrics.compute()
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

# ============================= Main Training Process ==============================
def main():
    
    # Prepare dataset
    print("Preparing dataset...")
    # HAM10000 dataset
    train_df, test_df = load_ham10000(config['metadata_path'], config['ham10000_image_root'])
    train_transforms, test_transforms = get_transforms()

    train_datasets = SkinData(train_df, transform=train_transforms)
    test_datasets = SkinData(test_df, transform=test_transforms)

    # OIA-ODIR dataset (commented out)
    # train_df, test_df = load_odir(
    #     train_dir = config['train_image_root'],
    #     test_dir  = config['test_image_root']
    # )
    # train_transforms, test_transforms = get_transforms()
    # train_datasets = ODIRDataFrame(train_df, transform=train_transforms, return_multihot=False)
    # test_datasets = ODIRDataFrame(test_df, transform=test_transforms, return_multihot=False)

    # Split into subsets
    client_datasets, client_dict = federated_dataset_split(
        train_datasets,
        num_clients=5,
        partition_mode="dirichlet",
        balance=True,
        dir_alpha=0.5,
        save_dir="odir_dirichlet_balanced",
        load_existing=True
    )

    # Model initialization
    print("Initializing model...")
    global_model = ResNet50(num_classes=num_classes, pretrained=True)
    global_model.to(device)

    global_test_loader = DataLoader(test_datasets, batch_size=config['batch_size'], shuffle=False)

    # Metrics collection
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
            )
        clients.append(client)
        dataset_sizes.append(len(client_datasets[idx]))

    # Training loop
    print("Starting model training...")
    for epoch in range(config['epochs']):
        start_time = time.time()
        w_locals = []

        for idx, client in enumerate(clients):
            loss, w, kappa, f1, auc, final, accuracy, precision, recall = client.train()
            w_locals.append(w)

        # Update learning rate scheduler
        for idx, client in enumerate(clients):
            client.update_scheduler()

        # FedAvg
        if (epoch + 1) % config['fedavg_num'] == 0:
            print("FedAvg...")
            w_global = federated_avg(w_locals, dataset_sizes)

            # Update global model parameters
            global_model.load_state_dict(w_global)
            # Evaluate with global model
            global_loss, global_kappa, global_f1, global_auc, global_final, global_acc, global_precision, global_recall = evaluate_global_model(global_model, global_test_loader, device)
            # Record global evaluation results
            test_metrics['epoch'].append(epoch+1)
            test_metrics['loss'].append(global_loss)
            test_metrics['accuracy'].append(global_acc)
            test_metrics['kappa'].append(global_kappa)
            test_metrics['f1'].append(global_f1)
            test_metrics['auc'].append(global_auc)
            test_metrics['final'].append(global_final)
            test_metrics['precision'].append(global_precision)
            test_metrics['recall'].append(global_recall)
            print(f"Epoch {epoch+1}/{config['epochs']}")
            print(f"Global model evaluation - Loss: {global_loss:.4f}, Kappa: {global_kappa:.4f}, F1: {global_f1:.4f}, "
                  f"AUC: {global_auc:.4f}, Final Score: {global_final:.4f}, Accuracy: {global_acc:.4f}, "
                  f"Precision: {global_precision:.4f}, Recall: {global_recall:.4f}")

            # Distribute global model parameters to all clients
            for idx, client in enumerate(clients):
                client.net.load_state_dict(w_global)

        # Training time per epoch
        end_time = time.time()
        print(f"Epoch {epoch+1} training time: {end_time - start_time:.2f} seconds")

    # Save results
    os.makedirs('results', exist_ok=True)
    file_name = config['resultfile']
    test_metrics_df = pd.DataFrame(test_metrics)
    with pd.ExcelWriter(file_name) as writer:
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