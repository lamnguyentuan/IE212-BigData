"""
Load Model Logic.
"""

import torch
import sys
from pathlib import Path
import yaml
from huggingface_hub import hf_hub_download

# Paths
ROOT = Path(__file__).resolve().parents[3]
# Ensure we can import offline_training
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from offline_training.models.multimodal_classifier import MultimodalClassifier

def load_finetuned_model(
    config_path: str = None, 
    checkpoint_path: str = None,
    device_str: str = "cpu"
) -> MultimodalClassifier:
    
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    print(f"Loading model on {device}...")

    # 1. Config
    # If not provided, try to find default finetune config
    if config_path is None:
        config_path = ROOT / "offline_training/finetune/finetune_config.yaml"
    
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    
    model_cfg = cfg["model"]
    
    # 2. Initialize Model (2-class for finetuned)
    # Ensure num_classes matches what is in finetune config (should be 2)
    # Note: Phase 5/6 we replaced head. Here we init fresh but need to match architecture.
    model = MultimodalClassifier(
        video_dim=model_cfg["video_dim"],
        audio_dim=model_cfg["audio_dim"],
        text_dim=model_cfg["text_dim"],
        meta_dim=model_cfg["meta_dim"],
        fusion_dim=model_cfg["fusion_dim"],
        num_classes=model_cfg["num_classes"],
        dropout=model_cfg["dropout"]
    )
    
    # 3. Load Weights
    # Priority:
    #   a) Local checkpoint path provided
    #   b) Default local artifact: offline_training/artifacts/finetune/finetuned_best.pt
    #   c) Pull from HF Hub (funa21/tiktok-vn-finetune)
    
    if checkpoint_path is None:
        local_default = ROOT / "offline_training/artifacts/finetune/finetuned_best.pt"
        if local_default.exists():
            checkpoint_path = str(local_default)
        else:
            try:
                print("Local checkpoint not found, downloading from HF Hub...")
                checkpoint_path = hf_hub_download(repo_id="funa21/tiktok-vn-finetune", filename="pytorch_model.bin") 
                # Note: PyTorchModelHubMixin usually saves as pytorch_model.bin or similar.
                # If our custom push used a different name, we might need to adjust.
                # Standard save_pretrained uses config.json + pytorch_model.bin
                # But our custom push in Phase 6 used push_to_hub(..., commit_message...) on class inheriting mixin.
                # It should be standard.
            except Exception as e:
                print(f"Failed to download from Hub: {e}")
                
    if checkpoint_path and Path(checkpoint_path).exists():
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"Loaded weights from {checkpoint_path}")
    else:
        print("WARNING: Using random weights!")

    model.to(device)
    model.eval()
    return model
