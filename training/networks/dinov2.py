import torch
import torch.nn as nn
from transformers import AutoModel
from metrics.registry import BACKBONE

@BACKBONE.register_module(module_name="dinov2")
class DINOv2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_classes = config["num_classes"]
        print("Initializing DINOv2Backbone with config", config)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model_name = config.get("model_name", "./abc") # make sure config has

        self.vision_model = AutoModel.from_pretrained(model_name, local_files_only=True)

        hidden = self.vision_model.config.hidden_size
        print("hidden_size=", hidden)

        self.last_layer = nn.Linear(hidden, self.num_classes)

        # Optionally freeze backbone
        if config.get("freeze_backbone", False):
            for p in self.vision_model.parameters():
                p.requires_grad = False

        # Optional PEFT (LoRA or LN-tuning). LN-tuning example similar in spirit to your CLIP LN tuning:
        if config.get("peft", False):
            from peft import get_peft_model, LoraConfig

            # LoRA tends to work better than LN-only for ViTs when you have lots of data.
            # You can tune target_modules depending on HF naming. Commonly "query"/"key"/"value"/"dense" exist.
            lora_config = LoraConfig(
                r=config.get("lora_r", 8),
                lora_alpha=config.get("lora_alpha", 16),
                lora_dropout=config.get("lora_dropout", 0.05),
                bias="none",
                target_modules=config.get(
                    "target_modules",
                    ["query", "key", "value", "dense", "fc1", "fc2"]
                ),
            )
            
            print("lora_config =", LoraConfig.to_dict(lora_config))
            self.vision_model = get_peft_model(self.vision_model, lora_config)

            total = 0
            trainable = 0
            for _, p in self.vision_model.named_parameters():
                total += p.numel()
                if p.requires_grad:
                    trainable += p.numel()
            print(f"Trainable params (backbone): {trainable}/{total} ({trainable/total:.2%})")

        self.count_parameters()

    def count_parameters(self):
        total_params = sum(p.numel() for p in self.parameters())
        backbone_params = sum(p.numel() for p in self.vision_model.parameters())
        tunable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params}")
        print(f"Total backbone parameters: {backbone_params}")
        print(f"Tunable parameters: {tunable_params}")
        print(f"Non-tunable parameters: {total_params - tunable_params}")

    def features(self, pixel_values):
        """
        pixel_values: torch.FloatTensor of shape [B, 3, H, W]
        """
        out = self.vision_model(pixel_values=pixel_values)

        # Prefer pooler_output if present; else use CLS token
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            # print("Using pooler_output as features")
            feats = out.pooler_output
        else:
            # print("Using CLS token as features")
            feats = out.last_hidden_state[:, 0]  # CLS token

        return feats  # shape [B, hidden_size]

    def classifier(self, feats):
        return self.last_layer(feats)

    def forward(self, pixel_values):
        feats = self.features(pixel_values)
        logits = self.classifier(feats)
        return logits
