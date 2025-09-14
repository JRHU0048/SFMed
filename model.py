import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

# Complete model - Multi-label classification
class ResNet50MultiLabel(nn.Module):
    """ResNet model for multi-label classification"""
    def __init__(self, num_classes=8, pretrained=True):
        super(ResNet50MultiLabel, self).__init__()
        # Load pretrained model
        self.base_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        # Replace classification head
        self.base_model.fc = nn.Identity()
        self.classifiers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2048, 256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, 1)
            ) for _ in range(num_classes)
        ])

    def forward(self, x, return_ssl=False):
        features = self.base_model(x)
        features = torch.flatten(features, 1)
        outputs = [classifier(features) for classifier in self.classifiers]
        return torch.cat(outputs, dim=1)  # [batch_size, num_classes]
    

# Split model - client side
class ResNet50_client_side(nn.Module):
    """Client-side model (first half of ResNet50)"""
    def __init__(self):
        super(ResNet50_client_side, self).__init__()
        base_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        
        # Extract the first half of ResNet50 (up to layer2)
        self.features = nn.Sequential(
            base_model.conv1,
            base_model.bn1,
            base_model.relu,
            base_model.maxpool,
            base_model.layer1,
            # base_model.layer2
        )
        
    def forward(self, x):
        return self.features(x)

class ResNet50_server_side(nn.Module):
    """Server-side model (second half of ResNet50 + multi-label classification head)"""
    def __init__(self, num_classes=8):
        super(ResNet50_server_side, self).__init__()
        base_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

        # Extract the second half of ResNet50 (after layer3)
        self.features = nn.Sequential(
            base_model.layer2,
            base_model.layer3,
            base_model.layer4,
            base_model.avgpool
        )

        # Multiple binary classification heads
        self.classifiers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2048, 256),
                nn.ReLU(),
                nn.Dropout(0.2),  # Add Dropout to classification head
                nn.Linear(256, 1)
            ) for _ in range(num_classes)
        ])

    def forward(self, x):
        features = self.features(x)
        features = torch.flatten(features, 1)
        outputs = [classifier(features) for classifier in self.classifiers]
        return torch.cat(outputs, dim=1)  # [batch_size, num_classes]
    
class ResNet50_server_side_ham(nn.Module):
    """Server-side model (second half of ResNet50 for HAM10000)"""
    def __init__(self, num_classes=7):
        super(ResNet50_server_side_ham, self).__init__()
        base_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

        # Extract the second half of ResNet50 (after layer3)
        self.features = nn.Sequential(
            base_model.layer2,
            base_model.layer3,
            base_model.layer4,
            base_model.avgpool
        )

        self.classifier = nn.Sequential(
            nn.Linear(2048, 256),  # Input 2048-dim features, output 256-dim
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)  # Output num_classes
        )

    def forward(self, x):
        features = self.features(x)  # Extract features
        features = torch.flatten(features, 1)  # Flatten features
        outputs = self.classifier(features)  # Classification head
        return outputs  # [batch_size, num_classes]
    

# Complete model - Multi-class classification
class ResNet50(nn.Module):
    """ResNet model for multi-class classification"""
    def __init__(self, num_classes=7, pretrained=True):
        super(ResNet50, self).__init__()
        # Load pretrained model
        self.base_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        # Replace classification head
        self.base_model.fc = nn.Identity()
        self.classifier = nn.Sequential(
            nn.Linear(2048, 256),  # Input 2048-dim features, output 256-dim
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)  # Output num_classes
        )

    def forward(self, x, return_ssl=False):
        features = self.base_model(x)  # Extract features
        features = torch.flatten(features, 1)  # Flatten features
        outputs = self.classifier(features)  # Classification head
        return outputs  # [batch_size, num_classes]