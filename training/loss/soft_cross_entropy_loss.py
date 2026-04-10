import torch
import torch.nn as nn
import torch.nn.functional as F
from .abstract_loss_func import AbstractLossClass
from metrics.registry import LOSSFUNC

@LOSSFUNC.register_module(module_name="soft_cross_entropy")
class SoftCrossEntropyLoss(AbstractLossClass):
    def __init__(self):
        super().__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()  # Binary Cross-Entropy Loss

    def forward(self, inputs, targets):
        """
        Computes the binary cross-entropy (BCE) loss.

        Args:
            inputs: A PyTorch tensor of size (batch_size, num_classes) containing raw logits.
            targets: A PyTorch tensor of size (batch_size, num_classes) containing target labels.

        Returns:
            A scalar tensor representing the BCE loss.
        """

        # Convert targets to one-hot encoding if necessary
        if targets.dim() == 1 or targets.shape[1] != inputs.shape[1]:
            if not torch.is_tensor(targets):
                targets = torch.tensor(targets, dtype=torch.long)
            else:
                targets = targets.long()

            num_classes = inputs.size(1)
            targets_one_hot = F.one_hot(targets, num_classes=num_classes).float()
        else:
            targets_one_hot = targets.float()

        # Ensure targets and inputs have the same shape
        if targets_one_hot.shape != inputs.shape:
            raise ValueError(f"Shape mismatch: inputs shape {inputs.shape}, targets shape {targets_one_hot.shape}")

        # Apply sigmoid activation to inputs and Compute BCE loss
        #probs = torch.sigmoid(inputs)
        #loss = self.bce_loss(probs, targets_one_hot)

        # Use BCEWithLogitsLoss to combine sigmoid activation and BCE loss
        loss = self.bce_loss(inputs, targets_one_hot)

        return loss
