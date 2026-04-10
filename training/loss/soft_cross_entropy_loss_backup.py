import torch
import torch.nn as nn
import torch.nn.functional as F
from .abstract_loss_func import AbstractLossClass
from metrics.registry import LOSSFUNC

# @LOSSFUNC.register_module(module_name="soft_cross_entropy")
class SoftCrossEntropyLoss_backup(AbstractLossClass):
    def __init__(self):
        super().__init__()

    def forward(self, inputs, targets):
        """
        Computes the soft cross-entropy loss.

        Args:
            inputs: A PyTorch tensor of size (batch_size, num_classes) containing the predicted scores.
            targets: A PyTorch tensor of size (batch_size, num_classes) containing the soft labels.

        Returns:
            A scalar tensor representing the soft cross-entropy loss.
        """

        # print(f"inputs shape: {inputs.shape}")
        # print(f"targets shape: {targets.shape}")
        # print(f"inputs: {inputs}")
        # print(f"targets: {targets}")

        # Ensure targets are in the same shape as inputs
        # if targets.shape != inputs.shape:
        #     raise ValueError(f"Shape mismatch: inputs shape {inputs.shape}, targets shape {targets.shape}")

        # # Apply log softmax to the inputs
        # log_probs = F.log_softmax(inputs, dim=-1)
        
        # # Compute the soft cross-entropy loss
        # loss = -(targets * log_probs).sum(dim=-1).mean()

        # return loss

        # Check if targets need to be converted to one-hot encoding
        if targets.dim() == 1 or targets.shape[1] != inputs.shape[1]:
            # Ensure targets are integer tensor
            if not torch.is_tensor(targets):
                targets = torch.tensor(targets, dtype=torch.long)
            else:
                targets = targets.long()
            
            # Convert targets to one-hot encoding
            num_classes = inputs.size(1)
            targets_one_hot = F.one_hot(targets, num_classes=num_classes).float()
        else:
            # Assume targets are already in one-hot format
            targets_one_hot = targets.float()

        # Ensure targets are in the same shape as inputs
        if targets_one_hot.shape != inputs.shape:
            raise ValueError(f"Shape mismatch: inputs shape {inputs.shape}, targets shape {targets_one_hot.shape}")

        # Apply log softmax to the inputs
        log_probs = F.log_softmax(inputs, dim=-1)
        
        # Compute the soft cross-entropy loss
        loss = -(targets_one_hot * log_probs).sum(dim=-1).mean()

        return loss