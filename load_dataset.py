from torchvision import transforms
from torch.utils.data import Dataset
import os
from PIL import Image
import torch
import yaml

# Load parameters
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)


class MultiLabelODIRDataset(Dataset):
    """Multi-label ophthalmic disease classification dataset"""
    UNIVERSAL_CLASS_NAMES = ['A', 'C', 'D', 'G', 'H', 'M', 'N', 'O']
    
    def __init__(self, image_root, transform=None):
        """
        Args:
            image_root: Path to image directory
            transform: Data augmentation transforms
        """
        self.image_root = image_root
        self.transform = transform
        
        self.image_paths = []
        self.labels = []
        
        for root, _, files in os.walk(image_root):
            class_name = os.path.basename(root)
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.image_paths.append(os.path.join(root, file))
                    self.labels.append([class_name])  # Single label list
        
        self.class_names = self.UNIVERSAL_CLASS_NAMES
        self.num_classes = len(self.class_names)
        
        # Convert labels to multi-hot encoding
        self.multihot_labels = []
        for labels in self.labels:
            multihot = [1 if cls in labels else 0 for cls in self.class_names]
            self.multihot_labels.append(multihot)
        
        print(f"Dataset loaded: {len(self.image_paths)} images, {self.num_classes} classes")
        print(f"Class list: {self.class_names}")
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        labels = torch.FloatTensor(self.multihot_labels[idx])
        return {
            'image': img,
            'labels': labels
        }

import os
from glob import glob
import pandas as pd

DEFAULT_ODIR_CLASSES = ['A','C','D','G','H','M','N','O']

def _scan_dir_to_records(data_dir, class_names):
    """
    Scan images organized by class_name subfolders under data_dir,
    Return record list: [{"image_id":..., "path":..., "label":..., "label_idx":...}, ...]
    """
    records = []
    class_to_idx = {c: i for i, c in enumerate(class_names)}
    for cls in class_names:
        cls_dir = os.path.join(data_dir, cls)
        if not os.path.isdir(cls_dir):
            continue
        patterns = [os.path.join(cls_dir, '*.jpg'),
                    os.path.join(cls_dir, '*.jpeg'),
                    os.path.join(cls_dir, '*.png')]
        img_paths = []
        for p in patterns:
            img_paths.extend(glob(p))
        for p in img_paths:
            image_id = os.path.splitext(os.path.basename(p))[0]
            records.append({
                "image_id": image_id,
                "path": p,
                "label": cls,
                "label_idx": class_to_idx[cls]
            })
    return records

def load_odir(train_dir, test_dir, class_names=None):
    """
    Load ODIR dataset from two separate paths (train and test)
    Returns (train_df, test_df), both with columns: ['image_id', 'path', 'label', 'label_idx']
    train_dir and test_dir should be organized by class subfolders (e.g. train_dir/A/*.jpg)
    """
    class_names = class_names or DEFAULT_ODIR_CLASSES

    train_records = _scan_dir_to_records(train_dir, class_names)
    if len(train_records) == 0:
        raise RuntimeError(f"No training images found under {train_dir}. Check directory structure.")

    test_records = _scan_dir_to_records(test_dir, class_names)
    if len(test_records) == 0:
        raise RuntimeError(f"No testing images found under {test_dir}. Check directory structure.")

    train_df = pd.DataFrame(train_records)
    test_df = pd.DataFrame(test_records)

    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)



class ODIRDataFrame(Dataset):
    """
    DataFrame -> Dataset wrapper for compatibility with upstream code.
    - df: must contain 'path' and 'label' or 'label_idx' columns
    - return_multihot: if True, returns multi-hot labels (float tensor); else returns integer label (torch.long)
    - self.targets is set as integer list for partitioner
    """
    def __init__(self, df, transform=None, class_names=None, return_multihot=False):
        self.df = df.reset_index(drop=True).copy()
        self.transform = transform
        self.class_names = class_names or DEFAULT_ODIR_CLASSES
        self.class_to_idx = {c:i for i,c in enumerate(self.class_names)}
        if 'label_idx' in self.df.columns:
            self.targets = [int(x) for x in self.df['label_idx'].tolist()]
        else:
            self.targets = [int(self.class_to_idx[l]) for l in self.df['label'].tolist()]

        self.return_multihot = return_multihot
        self.num_classes = len(self.class_names)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row['path']).convert('RGB')
        if self.transform:
            img = self.transform(img)

        label_idx = int(self.targets[idx])
        if self.return_multihot:
            lab = torch.zeros(self.num_classes, dtype=torch.float32)
            lab[label_idx] = 1.0
            labels = lab
        else:
            labels = torch.tensor(label_idx, dtype=torch.long)

        return {'image': img, 'labels': labels}


def fedavg_prepare_datasets():
    """Prepare training and test datasets"""
    train_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.1), ratio=(0.3, 3.3))
    ])
    
    test_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    train_datasets = []
    for idx in range(config['num_clients']):
        train_dataset = MultiLabelODIRDataset(
            config['fedavg_train_image_root']+f'/{idx}', 
            transform=train_transforms
        )
        train_datasets.append(train_dataset)

    offsite_test_datasets = MultiLabelODIRDataset(
        config['offsite_test_image_root'],
        transform=test_transforms
    )

    onsite_test_datasets = MultiLabelODIRDataset(
        config['onsite_test_image_root'],
        transform=test_transforms
    )

    return train_datasets, offsite_test_datasets, onsite_test_datasets


def sl_prepare_datasets():
    """Prepare training and test datasets"""
    train_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.1), ratio=(0.3, 3.3))
    ])
    
    test_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    train_datasets = []
    for idx in range(config['num_clients']):
        train_dataset = MultiLabelODIRDataset(
            config['split_train_image_root']+f'/{idx}', 
            transform=train_transforms
        )
        train_datasets.append(train_dataset)
    
    offsite_test_datasets = MultiLabelODIRDataset(
        config['offsite_test_image_root'],
        transform=test_transforms
    )

    onsite_test_datasets = MultiLabelODIRDataset(
        config['onsite_test_image_root'],
        transform=test_transforms
    )

    return train_datasets, offsite_test_datasets, onsite_test_datasets


# =================================== HAM10000 dataset =============================================
import random
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import random_split
from glob import glob 
import pandas as pd


class SkinData(Dataset):
    """Skin disease dataset loader"""
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform
        self.lesion_type = {
            'nv': 0, 'mel': 1, 'bkl': 2, 'bcc': 3,
            'akiec': 4, 'vasc': 5, 'df': 6
        }
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        img_path = self.df.iloc[index]['path']
        image = Image.open(img_path).resize((224, 224))
        label = self.lesion_type[self.df.iloc[index]['dx']]
        
        if self.transform:
            image = self.transform(image)
            
        return {
            'image': image,
            'labels': torch.tensor(label, dtype=torch.long)
        }

def get_transforms():
    """Get data augmentation and preprocessing transforms"""
    train_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.Pad(3),
        transforms.RandomRotation(10),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                            std=[0.229, 0.224, 0.225])
    ])
    
    test_transforms = transforms.Compose([
        transforms.Pad(3),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])
    
    return train_transforms, test_transforms

def dataset_iid(dataset, num_users):
    total_samples = len(dataset)
    samples_per_client = total_samples // num_users
    remaining = total_samples % num_users
    lengths = [samples_per_client] * num_users
    if remaining > 0:
        lengths[-1] += remaining
    return random_split(dataset, lengths)

def load_ham10000(data_path, image_dir, test_size=0.1):
    """Load HAM10000 dataset"""
    df = pd.read_csv(data_path)
    imageid_path = {os.path.splitext(os.path.basename(x))[0]: x
                   for x in glob(os.path.join(image_dir, '*', '*.jpg'))}
    df['path'] = df['image_id'].map(imageid_path.get)
    train_df, test_df = train_test_split(df, test_size=test_size)
    return train_df.reset_index(), test_df.reset_index()


class DatasetSplit(Dataset):
    """Dataset split class for creating client-specific subsets"""
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label

def print_class_distribution(dataset, dataset_name="HAM10000"):
    """
    Print label and sample count for each class in the dataset
    
    Args:
        dataset: SkinData dataset object
        dataset_name: dataset name (for print)
    """
    class_names = {
        0: 'nv (Melanocytic nevi)',
        1: 'mel (Melanoma)',
        2: 'bkl (Benign keratosis-like lesions)',
        3: 'bcc (Basal cell carcinoma)',
        4: 'akiec (Actinic keratoses)',
        5: 'vasc (Vascular lesions)',
        6: 'df (Dermatofibroma)'
    }
    class_counts = {class_id: 0 for class_id in class_names.keys()}
    for _, label in dataset:
        if isinstance(label, torch.Tensor):
            label = label.item()
        class_counts[label] += 1
    print(f"\n{'='*30}")
    print(f"{dataset_name} Class Distribution")
    print('-'*30)
    for class_id, count in class_counts.items():
        print(f"{class_names[class_id]:<40}: {count:>5} samples ({count/len(dataset):.1%})")
    print(f"{'Total':<40}: {len(dataset):>5} samples")
    print('='*30)
# =================================== HAM10000 dataset =============================================


# =================================== MNIST dataset =============================================
import torch
from torch.utils.data import Dataset, random_split
from torchvision import datasets, transforms
from collections import Counter

class MNISTWrapper(Dataset):
    """Wrap MNIST to output dict format like SkinData"""
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        image, label = self.base_dataset[idx]
        return {'image': image, 'labels': label}

def get_transforms():
    """Get data augmentation and preprocessing transforms"""
    train_transforms = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    test_transforms = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    return train_transforms, test_transforms

def load_mnist(data_dir='./data'):
    """Load MNIST dataset"""
    train_transforms, test_transforms = get_transforms()
    train_dataset_raw = datasets.MNIST(root=data_dir, train=True, download=True, transform=train_transforms)
    test_dataset_raw = datasets.MNIST(root=data_dir, train=False, download=True, transform=test_transforms)
    train_dataset = MNISTWrapper(train_dataset_raw)
    test_dataset = MNISTWrapper(test_dataset_raw)
    return train_dataset, test_dataset

def dataset_iid(dataset, num_users):
    total_samples = len(dataset)
    samples_per_client = total_samples // num_users
    remaining = total_samples % num_users
    lengths = [samples_per_client] * num_users
    if remaining > 0:
        lengths[-1] += remaining
    return random_split(dataset, lengths)

def print_class_distribution(dataset, dataset_name="MNIST"):
    labels = [sample['label'] for sample in dataset]
    count_dict = Counter(labels)
    print(f"\n{'='*30}")
    print(f"{dataset_name} Class Distribution")
    print('-'*30)
    for label in range(10):
        print(f"Class {label:<2}: {count_dict[label]:>5} samples ({count_dict[label]/len(dataset):.1%})")
    print(f"{'Total':<8}: {len(dataset):>5} samples")
    print('='*30)
# =================================== MNIST dataset =============================================


# =================================== Fashion-MNIST dataset =============================================
import torch
from torch.utils.data import Dataset, random_split
from torchvision import datasets, transforms
from collections import Counter

class FMNISTWrapper(Dataset):
    """Wrap Fashion-MNIST to output {'image': tensor, 'labels': int}"""
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        image, label = self.base_dataset[idx]
        return {'image': image, 'labels': label}

def get_transforms():
    train_transforms = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    test_transforms = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    return train_transforms, test_transforms

def load_fmnist(data_dir='./data'):
    train_transforms, test_transforms = get_transforms()
    train_dataset_raw = datasets.FashionMNIST(root=data_dir, train=True, download=True, transform=train_transforms)
    test_dataset_raw = datasets.FashionMNIST(root=data_dir, train=False, download=True, transform=test_transforms)
    train_dataset = FMNISTWrapper(train_dataset_raw)
    test_dataset = FMNISTWrapper(test_dataset_raw)
    return train_dataset, test_dataset

def dataset_iid(dataset, num_users):
    total_samples = len(dataset)
    samples_per_client = total_samples // num_users
    remaining = total_samples % num_users
    lengths = [samples_per_client] * num_users
    if remaining > 0:
        lengths[-1] += remaining
    return random_split(dataset, lengths)

def print_class_distribution(dataset, dataset_name="Fashion-MNIST"):
    labels = [sample['labels'] for sample in dataset]
    count_dict = Counter(labels)
    print(f"\n{'='*30}")
    print(f"{dataset_name} Class Distribution")
    print('-'*30)
    for label in range(10):
        print(f"Class {label:<2}: {count_dict[label]:>5} samples ({count_dict[label]/len(dataset):.1%})")
    print(f"{'Total':<8}: {len(dataset):>5} samples")
    print('='*30)
# =================================== Fashion-MNIST dataset =============================================


# =================================== CIFAR-10 dataset =============================================
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from collections import Counter

class CIFAR10Wrapper(Dataset):
    """Wrap CIFAR-10 to output {'image': tensor, 'labels': int}"""
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        image, label = self.base_dataset[idx]
        return {'image': image, 'labels': label}

def get_cifar10_transforms():
    train_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    return train_transforms, test_transforms

def load_cifar10(data_dir='./data'):
    train_transforms, test_transforms = get_cifar10_transforms()
    train_dataset_raw = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_transforms)
    test_dataset_raw = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=test_transforms)
    train_dataset = CIFAR10Wrapper(train_dataset_raw)
    test_dataset = CIFAR10Wrapper(test_dataset_raw)
    return train_dataset, test_dataset

def print_cifar10_class_distribution(dataset, dataset_name="CIFAR-10"):
    labels = [sample['labels'] for sample in dataset]
    count_dict = Counter(labels)
    print(f"\n{'='*30}")
    print(f"{dataset_name} Class Distribution")
    print('-'*30)
    for label in range(10):
        print(f"Class {label:<2}: {count_dict[label]:>5} samples ({count_dict[label]/len(dataset):.1%})")
    print(f"{'Total':<8}: {len(dataset):>5} samples")
    print('='*30)
# =================================== CIFAR-10 dataset =============================================


# ========================= Dataset split function using fedlab =========================
import os
import torch
from torch.utils.data import Dataset, Subset
from fedlab.utils.dataset.partition import CIFAR10Partitioner
import pickle

def federated_dataset_split(dataset, 
                            num_clients=5, 
                            partition_mode="iid", 
                            balance=None, 
                            dir_alpha=0.3, 
                            num_shards=None, 
                            unbalance_sgm=0.5, 
                            seed=42,
                            save_dir="splits",
                            load_existing=True):
    """
    General federated dataset split function, supports saving and loading split results
    
    Args:
        dataset (Dataset): Any Dataset, as long as __getitem__ returns dict or (image, label)
        num_clients (int): number of clients
        partition_mode (str): split strategy, options: ["iid", "dirichlet", "shards"]
        balance (bool): whether each client has the same number of samples
        dir_alpha (float): Dirichlet distribution alpha
        num_shards (int): for shards partition
        unbalance_sgm (float): sample number difference for unbalanced split
        seed (int): random seed
        save_dir (str): directory to save split results
        load_existing (bool): whether to try loading existing split
    Returns:
        client_datasets (list of Subset): Dataset for each client
        client_dict (dict): client_id -> sample indices
    """
    os.makedirs(save_dir, exist_ok=True)
    split_file = os.path.join(save_dir, f"split_{partition_mode}_clients{num_clients}_alpha{dir_alpha}.pkl")

    if load_existing and os.path.exists(split_file):
        print(f"Loading existing split: {split_file}")
        with open(split_file, "rb") as f:
            client_dict = pickle.load(f)
    else:
        print(f"Generating new split: {partition_mode}")
        sample0 = dataset[0]
        if isinstance(sample0, dict):
            targets = [dataset[i]['labels'] for i in range(len(dataset))]
        else:
            targets = [dataset[i][1] for i in range(len(dataset))]

        if partition_mode == "iid":
            fed_partitioner = CIFAR10Partitioner(
                targets,
                num_clients=num_clients,
                balance=balance,
                partition="iid",
                unbalance_sgm=unbalance_sgm,
                seed=seed
            )
        elif partition_mode == "dirichlet":
            fed_partitioner = CIFAR10Partitioner(
                targets,
                num_clients=num_clients,
                balance=balance,
                partition="dirichlet",
                dir_alpha=dir_alpha,
                unbalance_sgm=unbalance_sgm,
                seed=seed
            )
        elif partition_mode == "shards":
            if num_shards is None:
                raise ValueError("num_shards must be set when using shards partition mode")
            fed_partitioner = CIFAR10Partitioner(
                targets,
                num_clients=num_clients,
                balance=balance,
                partition="shards",
                num_shards=num_shards,
                seed=seed
            )
        else:
            raise ValueError(f"Unsupported partition_mode: {partition_mode}")

        client_dict = fed_partitioner.client_dict

        with open(split_file, "wb") as f:
            pickle.dump(client_dict, f)
        print(f"Split result saved to {split_file}")

    client_datasets = [Subset(dataset, client_dict[i]) for i in range(num_clients)]

    return client_datasets, client_dict
# ========================= Dataset split function using fedlab =========================