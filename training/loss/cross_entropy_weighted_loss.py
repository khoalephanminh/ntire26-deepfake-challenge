import torch
import torch.nn.functional as F
from .abstract_loss_func import AbstractLossClass
from metrics.registry import LOSSFUNC


@LOSSFUNC.register_module(module_name="cross_entropy_weighted")
class CrossEntropyWeightedLoss(AbstractLossClass):
    def __init__(
        self,
        class_weights=None,
        num_real=None,
        num_fake=None,
        reduction="mean",
        label_smoothing=0.0,
        max_weight=None,   # optional clamp, e.g. 10.0
    ):
        super().__init__()
        self.reduction = reduction
        self.label_smoothing = float(label_smoothing)

        # If explicit weights provided, use them.
        if class_weights is not None:
            w = torch.as_tensor(class_weights, dtype=torch.float32)

        # Else compute from counts (balanced inverse-frequency)
        elif (num_real is not None) and (num_fake is not None):
            num_real = float(num_real)
            num_fake = float(num_fake)
            N = num_real + num_fake

            w_real = N / (2.0 * max(num_real, 1.0))  # class 0
            w_fake = N / (2.0 * max(num_fake, 1.0))  # class 1
            w = torch.tensor([w_real, w_fake], dtype=torch.float32)

            if max_weight is not None:
                w = torch.clamp(w, max=float(max_weight))

        else:
            w = None

        self.register_buffer("class_weights", w)
        print("Init CrossEntropyWeightedLoss weights=", None if w is None else w.tolist())
        print("Fake/Real weight ratio=", None if w is None else (w[0] / w[1]).item())

    def forward(self, inputs, targets):
        targets = targets.long()
        w = self.class_weights
        if w is not None and w.device != inputs.device:
            w = w.to(inputs.device)

        return F.cross_entropy(
            inputs,
            targets,
            weight=w,
            reduction=self.reduction,
            label_smoothing=self.label_smoothing,
        )
