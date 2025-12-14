"""
Load Model Logic.
"""

import torch
import sys
from pathlib import Path
import yaml
from huggingface_hub import hf_hub_download

# Paths
ROOT = Path(__file__).resolve().parents[2]
# Ensure we can import offline_training
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from common.models.multimodal_classifier import MultimodalClassifier

def load_finetuned_model(
    config_path: str = None, 
    checkpoint_path: str = None,
    device_str: str = "cpu"
) -> MultimodalClassifier:
    
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    print(f"Loading model on {device}...")

    # 1. Config
    # If not provided, try to find default finetune config
    import os
    if config_path is None:
        config_path = os.getenv("CONFIG_PATH")
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
        dropout=model_cfg["dropout"],
        fusion_heads=model_cfg.get("fusion_heads", 8), # ✨
        fusion_layers=model_cfg.get("fusion_layers", 2) # ✨
    )
    
    # 3. Load Weights
    # Priority:
    #   a) Local checkpoint path provided
    #   b) Default local artifact: offline_training/artifacts/finetune/finetuned_best.pt
    #   c) Pull from HF Hub (funa21/tiktok-vn-finetune)
    
    if checkpoint_path is None:
        checkpoint_path = os.getenv("CHECKPOINT_PATH")

    if checkpoint_path is None:
        # Check if user wants a specific HF Repo
        hf_repo = os.getenv("HF_HUB_REPO", "funa21/tiktok-vn-finetune")
        
        # Look in local model_store first only if env var not forcing HF
        local_default = ROOT / "model-serving/model_store/finetuned_best.pt"
        
        # If explicitly requested HF or local doesn't exist
        force_hf = os.getenv("FORCE_HF_DOWNLOAD", "false").lower() == "true"
        
        if not force_hf and local_default.exists():
            checkpoint_path = str(local_default)
            print(f"Using local checkpoint: {checkpoint_path}")
        else:
            try:
                print(f"Downloading model from HF Hub: {hf_repo}...")
                checkpoint_path = hf_hub_download(repo_id=hf_repo, filename="model.safetensors") 
            except Exception as e:
                print(f"Failed to download from Hub: {e}")
                # Fallback to local if exists and we tried HF
                if local_default.exists():
                    print("Fallback to local checkpoint.")
                    checkpoint_path = str(local_default)
                
    if checkpoint_path and Path(checkpoint_path).exists():
        if str(checkpoint_path).endswith(".safetensors"):
             from safetensors.torch import load_file
             state_dict = load_file(checkpoint_path, device=device_str)
        else:
             state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"Loaded weights from {checkpoint_path}")
    else:
        print("WARNING: Using random weights!")

    model.to(device)
    model.eval()
    return model
