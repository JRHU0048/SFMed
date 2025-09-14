import torch
import torch.nn as nn
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
from model import ResNet50_client_side, ResNet50_server_side_ham
from load_dataset import load_ham10000, get_transforms, SkinData, dataset_iid, federated_dataset_split, load_odir, ODIRDataFrame, load_fmnist
import time

# Load parameters
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Device
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

num_classes = 8

# ============================= Server Class ==============================
class SLServer:
    """Server class"""
    def __init__(self, server_model, lr, device, client_id=0):
        self.device = device
        self.client_id = client_id
        self.server_model = copy.deepcopy(server_model).to(self.device)

        # Loss functions
        self.criterion = LabelSmoothingCrossEntropyLoss(smoothing=0.1)   # Label smoothing for training
        self.criterion_eval = nn.CrossEntropyLoss()  # Standard loss for evaluation

        # Optimizer and learning rate warmup
        self.warmup_epochs = 0  # No warmup
        self.current_epoch = 0  # Track current epoch
        self.lr = lr       # Initial learning rate (warmup target)
        self.optimizer = torch.optim.Adam(
            self.server_model.parameters(), 
            lr = float(config['lr']),
            weight_decay = float(config['weight_decay'])
        )
        
        # Learning rate scheduler
        self.scheduler = self._get_scheduler(lr)
        
    def train(self, client_fx, labels):
        """Train server model (with domain adaptation)"""
        self.server_model.train()
        
        client_fx = client_fx.to(self.device)
        labels = labels.to(self.device)
        
        # Forward
        class_outputs = self.server_model(client_fx)
        
        # Classification loss
        class_loss = self.criterion(class_outputs, labels)
        
        # Backward
        self.optimizer.zero_grad()
        class_loss.backward()
        self.optimizer.step()
        
        # Return client gradient
        client_dfx = client_fx.grad.clone().detach()
        return client_dfx
    
    def evaluate(self, client_fx, labels, metrics, isOnsite):
        """Evaluate server model"""
        self.server_model.eval()
        
        with torch.no_grad():
            client_fx = client_fx.to(self.device)
            labels = labels.to(self.device)
            
            # Classification output
            class_outputs = self.server_model(client_fx)
            
            # Compute loss
            loss = self.criterion_eval(class_outputs, labels)
            
            # Update metrics
            metrics.update(class_outputs, labels)
            
            return loss.item()

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
        if self.scheduler is None:
            return
        
        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            if metric is not None:
                self.scheduler.step(metric)
        else:
            self.scheduler.step()
            
        # Print current learning rate
        current_lr = self.optimizer.param_groups[0]['lr']
        print(f"Server {self.client_id} current learning rate: {current_lr:.2e}")

# ============================= Client Class ==============================
class SLClient:
    """Client class"""
    def __init__(self, client_model, idx, lr, device, 
                 dataset_train, dataset_test_onsite):
        self.client_model = copy.deepcopy(client_model).to(device)
        self.idx = idx
        self.device = device
        self.lr = lr
        
        # Optimizer and learning rate warmup
        self.warmup_epochs = 0  # No warmup
        self.current_epoch = 0  # Track current epoch
        self.lr = lr       # Initial learning rate (warmup target)
        self.optimizer = torch.optim.Adam(
            self.client_model.parameters(), 
            lr=float(config['lr']),
            weight_decay=float(config['weight_decay'])
        )

        # Data loaders
        self.train_loader = DataLoader(dataset_train, batch_size=config['batch_size'], shuffle=True)
        self.test_loader = DataLoader(dataset_test_onsite, batch_size=config['batch_size'], shuffle=False)
        
        # Evaluation metrics
        self.metrics = ClassificationMetrics(num_classes=num_classes)
    
        # Learning rate scheduler
        self.scheduler = self._get_scheduler(lr)


    def train(self, server):
        """Client training"""
        self.client_model.train()
        print(f"Training set size: {len(self.train_loader.dataset)}")
        
        for batch in self.train_loader:
            images = batch['image'].to(self.device)
            labels = batch['labels'].to(self.device)
            self.optimizer.zero_grad()
            
            # Client forward
            activations = self.client_model(images)
            client_fx = activations.clone().detach().requires_grad_(True)
            
            # Server training and get gradient
            client_dfx = server.train(client_fx, labels)
            
            # Client backward
            activations.backward(client_dfx)
            self.optimizer.step()
        
        return self.client_model.state_dict()

    def evaluate(self, server):
        """Client evaluation"""
        self.client_model.eval()
        self.metrics.reset()
        total_loss = 0.0
        count = 0
        
        test_loader = self.test_loader 
        
        with torch.no_grad():
            for batch in test_loader:
                images = batch['image'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Client forward
                activations = self.client_model(images)
                
                # Server evaluation
                loss = server.evaluate(activations, labels, self.metrics, isOnsite=True)
                total_loss += loss
                count += 1
        
        # Compute average loss and metrics
        avg_loss = total_loss / count
        metrics_results = self.metrics.compute()
        
        print(f"test dataset - Client {self.idx} | "
              f"Loss: {avg_loss:.4f} | Kappa: {metrics_results['kappa']:.4f} | "
              f"F1: {metrics_results['f1']:.4f} | AUC: {metrics_results['auc']:.4f} | "
              f"Final: {metrics_results['final_score']:.4f}")
        
        return avg_loss, metrics_results

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
        if self.scheduler is None:
            return
        
        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            if metric is not None:
                self.scheduler.step(metric)
        else:
            self.scheduler.step()
            
        # Print current learning rate
        current_lr = self.optimizer.param_groups[0]['lr']
        print(f"Client {self.idx} current learning rate: {current_lr:.2e}")

# ============================= Main Training Process ==============================
def main():
    
    # Prepare dataset
    print("Preparing dataset...")
    train_df, test_df = load_ham10000(config['metadata_path'], config['ham10000_image_root'])
    train_transforms, test_transforms = get_transforms()
    # Create datasets
    train_datasets = SkinData(train_df, transform=train_transforms)
    test_datasets = SkinData(test_df, transform=test_transforms)

    # odir dataset
    # train_df, test_df = load_odir(
    #     train_dir = config['train_image_root'],
    #     test_dir  = config['test_image_root']
    # )
    # train_transforms, test_transforms = get_transforms()
    # train_datasets = ODIRDataFrame(train_df, transform=train_transforms, return_multihot=False)
    # test_datasets = ODIRDataFrame(test_df, transform=test_transforms, return_multihot=False)

    client_datasets, client_dict = federated_dataset_split(
        train_datasets,
        num_clients=5,
        partition_mode="dirichlet",
        balance=True,
        dir_alpha=0.5,
        save_dir="odir_dirichlet_balanced",
        load_existing=True   # True means load if already saved
    )
    
    # ========================== Create client instances =========================== 
    clients = []
    for idx in range(config['num_clients']):
        print(f"Initializing client {idx} model...")
        client_model = ResNet50_client_side()
        client_model.to(device)

        # Create client instance 
        client = SLClient(
            client_model=client_model,
            idx=idx,
            lr=float(config['lr']),
            device=device,
            dataset_train=client_datasets[idx],
            dataset_test_onsite=test_datasets,
        )
        clients.append(client)
    # ========================================================================================

    # ====================================== 
    # Initialize server model
    print("Initializing server model...")
    global_server_model = ResNet50_server_side_ham(num_classes=num_classes)
    global_server_model.to(device)

    # Create server instance
    server = SLServer(global_server_model, float(config['lr']), device)
    # ======================================

    # Training metrics collection
    metrics = {
        'test': {'epoch': [], 'loss': [], 'kappa': [], 'f1': [], 'auc': [], 'final_score': [], 
                    'accuracy': [], 'precision': [], 'recall': []},
    }

    # Training loop
    print("Start training...")
    for epoch in range(config['epochs']):
        print(f"Epoch {epoch + 1}/{config['epochs']}")
        start_time = time.time()  # Record start time

        # Client training and evaluation
        test_results = {'loss': [], 'kappa': [], 'f1': [], 'auc': [], 'final_score': [], 'accuracy': [], 'precision': [], 'recall': []}

        for idx, client in enumerate(clients):
            # train
            weights = client.train(server)

            # validation
            loss, res = client.evaluate(server)
            test_results['loss'].append(loss)
            for k in ['kappa', 'f1', 'auc', 'final_score', 'accuracy', 'precision', 'recall']:
                test_results[k].append(res[k])

        # Training time per epoch
        end_time = time.time()
        print(f"Epoch {epoch+1} training time: {end_time - start_time:.2f} seconds")

        # Update server scheduler
        server.update_scheduler()

        # Update client schedulers (synchronized learning rate update is more stable)
        for client in clients:
            client.update_scheduler()
    
        # Record metrics
        metrics['test']['epoch'].append(epoch + 1)
        for k in ['loss', 'kappa', 'f1', 'auc', 'final_score', 'accuracy', 'precision', 'recall']:
            metrics['test'][k].append(np.mean(test_results[k]))
    

    # Save results
    print("Saving results...")
    os.makedirs('results', exist_ok=True)
    file_name = config['result_file']
    test_df = pd.DataFrame(metrics['test'])
    with pd.ExcelWriter(file_name) as writer:
        test_df.to_excel(writer, sheet_name='Test Metrics', index=False)
    print(f"Training finished! Results saved to {file_name}")

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