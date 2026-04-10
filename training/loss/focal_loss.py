import torch
import torch.nn as nn
import torch.nn.functional as F
from .abstract_loss_func import AbstractLossClass
from metrics.registry import LOSSFUNC


@LOSSFUNC.register_module(module_name="focal_loss")
class FocalLoss(AbstractLossClass):
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        """
        Focal Loss for classification tasks.
        
        Args:
            alpha: Scaling factor. Set to 1.0 by default to rely purely on gamma for hard-mining.
            gamma: The focusing parameter. Higher = more aggressive focus on hard examples (default: 2.0).
            reduction: 'mean', 'sum', or 'none'.
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Computes the focal loss.

        Args:
            inputs: A PyTorch tensor of size (batch_size, num_classes) containing the predicted logits.
            targets: A PyTorch tensor of size (batch_size) containing the ground-truth class indices.

        Returns:
            A scalar tensor representing the focal loss.
        """
        # 1. Compute standard Cross-Entropy Loss without reducing it yet
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # 2. Get the probability of the correct class (pt)
        # Because CE = -log(pt), we can get pt by taking exp(-CE)
        pt = torch.exp(-ce_loss)
        
        # 3. Apply the focal loss mathematical formula: alpha * (1 - pt)^gamma * CE
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        # 4. Apply reduction
        if self.reduction == 'mean':
            return torch.mean(focal_loss)
        elif self.reduction == 'sum':
            return torch.sum(focal_loss)
        else:
            return focal_loss