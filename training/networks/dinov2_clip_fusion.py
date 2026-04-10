import torch
import torch.nn as nn
from transformers import AutoModel, CLIPVisionModel
from metrics.registry import BACKBONE

@BACKBONE.register_module(module_name="dinov2_clip_fusion")
class DINOv2_CLIP_Fusion(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_classes = config["num_classes"]
        print("Initializing DINOv2 + CLIP-L Fusion Backbone with config", config)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # =========================================================
        # 1. INITIALIZE DINOv2 (Named 'vision_model' to match 9248 ckpt keys)
        # =========================================================
        dino_model_name = config.get("model_name", "models/dinov2-giant")
        self.vision_model = AutoModel.from_pretrained(dino_model_name, local_files_only=True)
        dino_hidden = self.vision_model.config.hidden_size
        print(f"DINOv2 hidden_size = {dino_hidden}")

        # Optional PEFT (LoRA) for DINO
        if config.get("peft", False):
            from peft import get_peft_model, LoraConfig
            lora_config = LoraConfig(
                r=config.get("lora_r", 32),
                lora_alpha=config.get("lora_alpha", 64),
                lora_dropout=config.get("lora_dropout", 0.15),
                bias="none",
                target_modules=config.get(
                    "target_modules",
                    ["query", "key", "value", "dense"]
                ),
            )
            self.vision_model = get_peft_model(self.vision_model, lora_config)
            print("Successfully attached LoRA layers to DINOv2.")

        # --- PHASE 1 WARMUP PROTECTION ---
        # If freeze_lora is True, we lock the entire DINO model (including LoRA weights)
        # so ONLY the new Fusion Head trains.
        if config.get("freeze_lora", False):
            for p in self.vision_model.parameters():
                p.requires_grad = False
            print("⚠️ PHASE 1 ACTIVE: DINOv2 and LoRA weights are FROZEN. Training Head only.")

        # =========================================================
        # 2. INITIALIZE CLIP-L (STRICTLY FROZEN)
        # =========================================================
        clip_model_name = config.get("clip_model_name", "models/clip-vit-large-patch14")
        self.clip_model = CLIPVisionModel.from_pretrained(clip_model_name, local_files_only=True)
        clip_hidden = self.clip_model.config.hidden_size
        print(f"CLIP-L hidden_size = {clip_hidden}")

        # Double-lock the gradients for CLIP
        for p in self.clip_model.parameters():
            p.requires_grad = False
        self.clip_model.eval()

        # =========================================================
        # 3. MULTI-MODAL FUSION HEAD
        # =========================================================
        fusion_dim = dino_hidden + clip_hidden
        
        # We use a 2-layer MLP to mix DINO and CLIP features
        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.GELU(),
            nn.Dropout(p=0.2), 
            nn.Linear(512, self.num_classes)
        )

        self.count_parameters()

    def count_parameters(self):
        total_params = sum(p.numel() for p in self.parameters())
        dino_params = sum(p.numel() for p in self.vision_model.parameters())
        clip_params = sum(p.numel() for p in self.clip_model.parameters())
        tunable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        print("\n=== PARAMETER COUNT ===")
        print(f"Total parameters: {total_params:,}")
        print(f"DINOv2 parameters: {dino_params:,}")
        print(f"CLIP-L parameters: {clip_params:,}")
        print(f"Tunable parameters: {tunable_params:,}")
        print("=======================\n")

    def features(self, pixel_values):
        # 1. DINO Features (Graph built only if freeze_lora is False)
        dino_out = self.vision_model(pixel_values=pixel_values)
        if hasattr(dino_out, "pooler_output") and dino_out.pooler_output is not None:
            dino_feats = dino_out.pooler_output
        else:
            dino_feats = dino_out.last_hidden_state[:, 0]  

        # 2. CLIP Features (No Memory Graph)
        with torch.no_grad():
            self.clip_model.eval() 
            clip_out = self.clip_model(pixel_values=pixel_values)
            if hasattr(clip_out, "pooler_output") and clip_out.pooler_output is not None:
                clip_feats = clip_out.pooler_output
            else:
                clip_feats = clip_out.last_hidden_state[:, 0] 

        # 3. Concat [B, 1536] and [B, 1024] -> [B, 2560]
        fused_feats = torch.cat((dino_feats, clip_feats), dim=-1)
        return fused_feats  

    def classifier(self, feats):
        return self.fusion_head(feats)

    def forward(self, pixel_values):
        feats = self.features(pixel_values)
        return self.classifier(feats)