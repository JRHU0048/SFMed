import torch
import torch.nn as nn
import torch.nn.functional as F

# Label Smoothing for BCEWithLogitsLoss
class LabelSmoothingBCEWithLogitsLoss(nn.Module):
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingBCEWithLogitsLoss, self).__init__()
        self.smoothing = smoothing
        self.bce = nn.BCEWithLogitsLoss()
    
    def forward(self, logits, labels):
        labels = labels * (1 - self.smoothing) + 0.5 * self.smoothing
        return self.bce(logits, labels)
    

class LabelSmoothingCrossEntropyLoss(nn.Module):
    """Label smoothing loss for single-label multi-class classification (based on CrossEntropyLoss)"""
    def __init__(self, smoothing=0.1, num_classes=None):
        super().__init__()
        if not 0.0 <= smoothing <= 1.0:
            raise ValueError("Smoothing must be in [0, 1]")
        self.smoothing = smoothing
        self.num_classes = num_classes  # Must specify total number of classes

    def forward(self, logits, labels):
        """
        Args:
            logits: Model outputs, shape [batch_size, num_classes]
            labels: Ground truth, shape [batch_size] (integer indices, single-label)
        """
        batch_size = logits.size(0)
        num_classes = self.num_classes or logits.size(1)  # Get number of classes from logits if not set
        
        # 1. Convert labels to one-hot encoding (shape [batch_size, num_classes])
        one_hot_labels = F.one_hot(labels, num_classes=num_classes).float()
        
        # 2. Label smoothing:
        # The probability at the original label position (1) becomes 1 - smoothing + smoothing/num_classes
        # Other positions (0) become smoothing/num_classes
        # Ensure the sum of probabilities for all classes is 1
        smoothed_labels = one_hot_labels * (1.0 - self.smoothing) + self.smoothing / num_classes
        
        # 3. Compute cross-entropy (manual calculation since CrossEntropyLoss does not support custom label distributions)
        log_probs = F.log_softmax(logits, dim=1)  # Apply softmax and take log
        # Cross-entropy = -sum(label_distribution * log(predicted_distribution))
        loss = -torch.sum(smoothed_labels * log_probs, dim=1).mean()
        return loss