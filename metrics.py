import torch
from sklearn.metrics import cohen_kappa_score, roc_auc_score, f1_score
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import f1_score, cohen_kappa_score, roc_auc_score, accuracy_score, precision_score, recall_score, confusion_matrix


class MultiLabelMetrics:
    """Multi-label classification metrics computation (following official evaluation metrics)"""
    
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.reset()
    
    def reset(self):
        """Reset all metrics"""
        self.y_true = []
        self.y_pred = []
        self.y_score = []
    
    def update(self, logits, labels):
        """Update metrics with new batch"""
        # Convert logits to probabilities
        probs = torch.sigmoid(logits).cpu().detach().numpy()
        # Convert logits to predicted classes (greater than 0.5 is 1)
        preds = (logits > 0.5).float().cpu().detach().numpy()
        # True labels
        true = labels.cpu().detach().numpy()
        
        self.y_true.append(true)
        self.y_pred.append(preds)
        self.y_score.append(probs)

    def compute(self):
        """Compute all metrics"""
        y_true = np.concatenate(self.y_true)
        y_pred = np.concatenate(self.y_pred)
        y_score = np.concatenate(self.y_score)
        
        # Initialize metrics
        metrics = {
            'kappa': 0.0,
            'f1': 0.0,
            'auc': 0.0,
            'final_score': 0.0
        }
        
        # Flatten all samples and all classes
        gt_flat = y_true.flatten()
        pr_flat = y_score.flatten()
        pred_flat = y_pred.flatten()
        
        # Kappa (using 0.5 as threshold)
        metrics['kappa'] = cohen_kappa_score(gt_flat, pred_flat)
        
        # F1 score (micro average)
        metrics['f1'] = f1_score(gt_flat, pred_flat, average='micro')
        
        # AUC
        metrics['auc'] = roc_auc_score(gt_flat, pr_flat)
        
        # Final score (average of three metrics)
        metrics['final_score'] = (metrics['kappa'] + metrics['f1'] + metrics['auc']) / 3.0
            
        return metrics

class ClassificationMetrics:
    def __init__(self, num_classes=7):
        self.num_classes = num_classes
        self.y_true = []
        self.y_pred = []
        self.y_prob = []  # probabilities after softmax for AUC

    def reset(self):
        self.y_true = []
        self.y_pred = []
        self.y_prob = []

    def update(self, outputs, labels):
        probs = F.softmax(outputs, dim=1)  # [B, num_classes]
        preds = torch.argmax(probs, dim=1)  # [B]

        self.y_true.extend(labels.detach().cpu().tolist())
        self.y_pred.extend(preds.detach().cpu().tolist())
        self.y_prob.extend(probs.detach().cpu().tolist())

    def compute(self):
        y_true = np.array(self.y_true)
        y_pred = np.array(self.y_pred)
        y_prob = np.array(self.y_prob)

        # Kappa coefficient
        kappa = cohen_kappa_score(y_true, y_pred)

        # F1 score
        f1 = f1_score(y_true, y_pred, average='macro')

        # AUC score
        auc = roc_auc_score(
            y_true=np.eye(self.num_classes)[y_true],
            y_score=y_prob,
            average='macro',
            multi_class='ovr'
        )

        # Accuracy
        accuracy = accuracy_score(y_true, y_pred)

        # Precision (Macro)
        precision = precision_score(y_true, y_pred, average='macro')

        # Recall (Macro)
        recall = recall_score(y_true, y_pred, average='macro')

        # Confusion Matrix
        conf_matrix = confusion_matrix(y_true, y_pred)

        # Final score (custom weight)
        final_score =  (f1 + kappa + auc + accuracy + precision + recall) / 6.0

        return {
            'kappa': kappa,
            'f1': f1,
            'auc': auc,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'confusion_matrix': conf_matrix,
            'final_score': final_score
        }