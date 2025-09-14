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
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Load parameters
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Device
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

num_classes = 7

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

        # Store server uncertainty (default 1.0)
        self.uncertainty_score = 1.0
        
    def train(self, client_fx, labels, all_servers):
        """Train server model (with domain adaptation)"""
        self.server_model.train()
        
        client_fx = client_fx.to(self.device)
        labels = labels.to(self.device)
        
        # Forward
        class_outputs = self.server_model(client_fx)
        
        # Classification loss
        class_loss = self.criterion(class_outputs, labels)

        # ===== Consistency loss among servers =====
        consistency_loss = 0.0
        other_servers = [s for s in all_servers if s.client_id != self.client_id]
        for other_server in other_servers:
            for (param_self, param_other) in zip(self.server_model.parameters(), other_server.server_model.parameters()):
                # Cosine Similarity (alignment)
                cos_sim = torch.nn.functional.cosine_similarity(param_self.view(-1), param_other.detach().view(-1), dim=0)
                consistency_loss += 1 - cos_sim  # The closer to 1, the more consistent

        consistency_loss = consistency_loss / len(other_servers)  # Average loss
        # ===== Consistency loss among servers =====

        # Total loss
        total_loss = class_loss + config['consistency_lambda'] * consistency_loss
        
        # Backward
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        # Return client gradient
        client_dfx = client_fx.grad.clone().detach()
        return client_dfx
    
    def evaluate(self, client_fx, labels, metrics):
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

    def compute_uncertainty(self, dataloader, client_model=None):
        """
        Compute server model uncertainty based on prediction entropy.
        If dataloader provides raw images, client_model must be provided.
        """
        self.server_model.eval()
        entropies = []
        with torch.no_grad():
            for batch in dataloader:
                images = batch['image'].to(self.device)

                # client forward to get intermediate features
                if client_model is None:
                    raise ValueError("If dataloader provides raw images, client_model must be provided")
                client_model.eval()
                client_fx = client_model(images)

                # server forward
                outputs = self.server_model(client_fx)
                probs = torch.softmax(outputs, dim=1)
                entropy = -(probs * torch.log(probs + 1e-12)).sum(dim=1).mean()
                entropies.append(entropy.item())

        self.uncertainty_score = float(np.mean(entropies))
        return self.uncertainty_score

    def avg_adaptive(self, all_servers, avg_keys):
        """
        Adaptive weighted averaging based on uncertainty.
        Weight = 1 / (uncertainty + eps), lower uncertainty gets higher weight.
        """
        eps = 1e-8
        all_state_dicts = [s.server_model.state_dict() for s in all_servers]

        # Compute weights
        uncertainties = [s.uncertainty_score for s in all_servers]
        weights = np.array([1.0 / (u + eps) for u in uncertainties])
        weights = weights / weights.sum()

        avg_state_dict = {}
        for key in all_state_dicts[0].keys():
            if any([key.startswith(k) for k in avg_keys]):
                # Weighted average
                weighted_params = 0
                for w, sd in zip(weights, all_state_dicts):
                    weighted_params += w * sd[key].float()
                avg_state_dict[key] = weighted_params
            else:
                avg_state_dict[key] = self.server_model.state_dict()[key]

        return avg_state_dict

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

    def avg(self, all_servers, avg_keys):
        """
        Average only specified layers, keep others unchanged.
        Args:
            all_servers: list of all server instances
            avg_keys: list of parameter name prefixes to average
        Returns:
            dict: updated full state_dict
        """
        all_state_dicts = [s.server_model.state_dict() for s in all_servers]
        avg_state_dict = {}

        for key in all_state_dicts[0].keys():
            # Check if this parameter needs to be averaged
            if any([key.startswith(k) for k in avg_keys]):
                # Average this layer
                avg_param = torch.stack([sd[key].float() for sd in all_state_dicts], dim=0).mean(dim=0)
                avg_state_dict[key] = avg_param
            else:
                # Not averaged, keep current server's own parameter
                avg_state_dict[key] = self.server_model.state_dict()[key]

        return avg_state_dict

# ============================= Client Class ==============================
class SLClient:
    """Client class"""
    def __init__(self, client_model, idx, lr, device, 
                 dataset_train, dataset_test_onsite, dataset_test_offsite=None):
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
            lr = float(config['lr']),
            weight_decay = float(config['weight_decay'])
        )

        # Data loaders
        self.train_loader = DataLoader(dataset_train, batch_size=config['batch_size'], shuffle=True)
        self.test_loader = DataLoader(dataset_test_onsite, batch_size=config['batch_size'], shuffle=False)
        self.offsite_test_loader = self.test_loader if dataset_test_offsite is None else DataLoader(dataset_test_offsite, batch_size=config['batch_size'], shuffle=False)
        
        # Evaluation metrics
        self.metrics = ClassificationMetrics(num_classes=num_classes)
    
        # Learning rate scheduler
        self.scheduler = self._get_scheduler(lr)

    def train(self, server, all_servers):
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
            client_dfx = server.train(client_fx, labels, all_servers)
            
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
                loss = server.evaluate(activations, labels, self.metrics)
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

    def extract_features(self, server, dataloader, max_samples=1000):
        """Extract client output features, flatten for t-SNE visualization"""
        self.client_model.eval()
        features, labels = [], []
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                images = batch['image'].to(self.device)
                lbls = batch['labels'].cpu().numpy()
                activations = self.client_model(images)
                activations = activations.view(activations.size(0), -1)
                features.append(activations.cpu().numpy())
                labels.append(lbls)
                if len(labels) * config['batch_size'] >= max_samples:
                    break
        return np.concatenate(features, axis=0), np.concatenate(labels, axis=0)


def extract_server_features(client, server, dataloader, max_samples_per_site=500):
    """
    Use server model to extract final features (not classification output).
    Note: dataloader provides raw images, need to use client model first, then server.
    """
    client.client_model.eval()
    server.server_model.eval()

    features_list = []
    labels_list = []
    collected = 0

    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(client.device)
            labels = batch['labels'].cpu().numpy()

            # client forward
            client_fx = client.client_model(images)

            # server forward to get final features (could be logits or last layer features)
            feats = server.server_model(client_fx)
            if feats.dim() > 2:
                feats = feats.view(feats.size(0), -1)
            feats = feats.cpu().numpy()

            features_list.append(feats)
            labels_list.append(labels)
            collected += feats.shape[0]
            if collected >= max_samples_per_site:
                break

    if len(features_list) == 0:
        return np.zeros((0,)), np.zeros((0,))

    features = np.concatenate(features_list, axis=0)[:max_samples_per_site]
    labels = np.concatenate(labels_list, axis=0)[:max_samples_per_site]
    return features, labels

def collect_all_server_features(clients, servers, loader_attr='train_loader', max_per_site=500):
    """
    Collect all server final features for each client
    Returns:
        features_all: (num_sites * max_per_site, dim)
        sites_all: (num_sites * max_per_site,)
        labels_all: (num_sites * max_per_site,)
    """
    feats_list = []
    sites_list = []
    labels_list = []

    for idx, (client, server) in enumerate(zip(clients, servers)):
        dataloader = getattr(client, loader_attr)
        feats, labs = extract_server_features(client, server, dataloader, max_samples_per_site=max_per_site)
        if feats.shape[0] == 0:
            continue
        feats_list.append(feats)
        sites_list.append(np.full((feats.shape[0],), idx, dtype=int))
        labels_list.append(labs)

    if len(feats_list) == 0:
        return np.zeros((0,0)), np.zeros((0,), dtype=int), np.zeros((0,), dtype=int)

    features_all = np.vstack(feats_list)
    sites_all = np.concatenate(sites_list)
    labels_all = np.concatenate(labels_list)
    return features_all, sites_all, labels_all

def plot_tsne_sites(features_before, sites_before, features_after, sites_after,
                    num_sites=None, save_path="results/tsne_sites.png", pca_dim=50, tsne_perplexity=30):
    """
    Visualization: left is before training (all site features), right is after training (all site features).
    features_*: (N, D) numpy array
    sites_*: (N,) int array site id (0..num_sites-1)
    """
    import os
    # Prevent OpenBLAS/OMP thread conflicts
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    # Combine for PCA (ensure both mappings in same subspace)
    X_all = np.vstack([features_before, features_after])
    pca = PCA(n_components=min(pca_dim, X_all.shape[0]-1))
    X_pca = pca.fit_transform(X_all)

    # t-SNE
    tsne = TSNE(n_components=2, random_state=42, init='pca', perplexity=tsne_perplexity)
    X_2d = tsne.fit_transform(X_pca)

    n_before = features_before.shape[0]
    X_before_2d = X_2d[:n_before]
    X_after_2d = X_2d[n_before:]

    # plotting
    plt.figure(figsize=(12,6))
    cmap = plt.get_cmap("tab10")
    unique_sites = np.unique(sites_before) if num_sites is None else np.arange(num_sites)

    # Left: before
    ax = plt.subplot(1,2,1)
    for s in unique_sites:
        idxs = np.where(sites_before == s)[0]
        if idxs.size == 0:
            continue
        ax.scatter(X_before_2d[idxs,0], X_before_2d[idxs,1], s=8, color=cmap(int(s) % 10), label=f"Site {s+1}", alpha=0.8)
    ax.legend(markerscale=2, fontsize='14', loc='best')
    ax.set_xticks([]); ax.set_yticks([])

    # Right: after
    ax2 = plt.subplot(1,2,2)
    for s in unique_sites:
        idxs = np.where(sites_after == s)[0]
        if idxs.size == 0:
            continue
        ax2.scatter(X_after_2d[idxs,0], X_after_2d[idxs,1], s=8, color=cmap(int(s) % 10), label=f"Site {s+1}", alpha=0.8)
    ax2.legend(markerscale=2, fontsize='14', loc='best')
    ax2.set_xticks([]); ax2.set_yticks([])

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved t-SNE visualization to {save_path}")

# ============================= Main Training Process ==============================
def main():

    # Prepare dataset
    print("Preparing dataset...")
    # ham dataset
    train_df, test_df = load_ham10000(config['metadata_path'], config['ham10000_image_root'])
    train_transforms, test_transforms = get_transforms()
    train_datasets = SkinData(train_df, transform=train_transforms)
    test_datasets = SkinData(test_df, transform=test_transforms)

    # odir dataset (commented out)
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
        balance=False,
        dir_alpha=0.5,
        save_dir="ham_dirichlet_unbalanced",
        load_existing=False
    )

    # ========================== Create client instances =========================== 
    clients = []
    for idx in range(config['num_clients']):
        print(f"Initializing client {idx} model...")
        client_model = ResNet50_client_side()
        client_model.to(device)

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

    # Create server instances
    servers = []
    for idx in range(config['num_clients']):
        server_model = ResNet50_server_side_ham(num_classes=num_classes).to(device)
        server = SLServer(server_model, float(config['lr']), device, client_id=idx)
        servers.append(server)

    # ======================================

    # Metrics collection
    metrics = {
        'offsite': {'epoch': [], 'loss': [], 'kappa': [], 'f1': [], 'auc': [], 'final_score': [], 
                    'accuracy': [], 'precision': [], 'recall': []},
        'onsite': {'epoch': [], 'loss': [], 'kappa': [], 'f1': [], 'auc': [], 'final_score': [], 
                   'accuracy': [], 'precision': [], 'recall': []}
    }

    # Extract initial features for each site before training
    print("Extracting features before training...")
    max_per_site = 500  

    features_before, sites_before, labels_before = collect_all_server_features(clients, servers,
                                                                            loader_attr='train_loader',
                                                                            max_per_site=max_per_site)

    # Training loop
    print("Start training...")
    for epoch in range(config['epochs']):
        print(f"Epoch {epoch + 1}/{config['epochs']}")
        start_time = time.time()

        offsite_results = {'loss': [], 'kappa': [], 'f1': [], 'auc': [], 'final_score': [], 'accuracy': [], 'precision': [], 'recall': []}

        for idx, client in enumerate(clients):
            server=servers[idx]

            # train
            weights = client.train(server=servers[idx], all_servers=servers)

            # validation
            loss, res = client.evaluate(server)
            offsite_results['loss'].append(loss)
            for k in ['kappa', 'f1', 'auc', 'final_score', 'accuracy', 'precision', 'recall']:
                offsite_results[k].append(res[k])

            # Current client final score
            final_result = offsite_results['accuracy'][-1]
            print(f"Client {idx} current Off-site accuracy: {final_result:.4f}")

        # After training, extract final server features for each site
        print("Extracting server features after training...")
        features_after, sites_after, labels_after = collect_all_server_features(clients, servers,
                                                                                loader_attr='train_loader',
                                                                                max_per_site=max_per_site)

        # Plot and save t-SNE comparison (colored by site)
        plot_tsne_sites(features_before, sites_before, features_after, sites_after,
                        num_sites=len(clients), save_path= f"results/tsne/sites_before_after_{epoch+1}.png",
                        pca_dim=7, tsne_perplexity=30)
    
        # Training time per epoch
        end_time = time.time()
        print(f"Epoch {epoch+1} training time: {end_time - start_time:.2f} seconds")

        # Update server schedulers
        for server in servers:
            server.update_scheduler()

        # Update client schedulers
        for client in clients:
            client.update_scheduler()

        # ================= Server aggregation =================
        print("Computing server uncertainty scores...")
        for idx, server in enumerate(servers):
            score = server.compute_uncertainty(
                clients[idx].offsite_test_loader,
                client_model=clients[idx].client_model
            )
            print(f"Server {idx} uncertainty score: {score:.4f}")

        print("Server adaptive parameter averaging based on uncertainty...")
        avg_keys = ['features.0','features.1','features.2']
        avg_params = servers[0].avg_adaptive(servers, avg_keys=avg_keys)
        for server in servers:
            server.server_model.load_state_dict(avg_params)
        # =================
    
        # Record metrics
        metrics['offsite']['epoch'].append(epoch + 1)
        for k in ['loss', 'kappa', 'f1', 'auc', 'final_score', 'accuracy', 'precision', 'recall']:
            metrics['offsite'][k].append(np.mean(offsite_results[k]))
        print("Current Off-site results:")
        print(f"Epoch {epoch + 1} | "
                f"Loss: {np.mean(offsite_results['loss']):.4f} | "
                f"Kappa: {np.mean(offsite_results['kappa']):.4f} | "  
                f"F1: {np.mean(offsite_results['f1']):.4f} | "
                f"AUC: {np.mean(offsite_results['auc']):.4f} | "
                f"Final Score: {np.mean(offsite_results['final_score']):.4f} | "
                f"Accuracy: {np.mean(offsite_results['accuracy']):.4f} | "
                f"Precision: {np.mean(offsite_results['precision']):.4f} | "
                f"Recall: {np.mean(offsite_results['recall']):.4f}")
    

    # Save results
    print("Saving results...")
    os.makedirs('results', exist_ok=True)
    file_name = config['resultfile']
    offsite_df = pd.DataFrame(metrics['offsite'])
    with pd.ExcelWriter(file_name) as writer:
        offsite_df.to_excel(writer, sheet_name='Off-site Metrics', index=False)
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