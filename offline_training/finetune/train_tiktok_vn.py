"""
Finetuning Script for TikTok VN (2-class classification).
"""

import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, recall_score

# Paths
ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT))

from offline_training.utils.config_loader import load_yaml
from offline_training.models.multimodal_classifier import MultimodalClassifier
from offline_training.datasets.multimodal_dataset import MultimodalDataset
from offline_training.preprocessing.utils.logging_utils import get_logger

logger = get_logger("finetune-tiktok")

def train():
    # 1. Load Config
    config_path = Path(__file__).parent / "finetune_config.yaml"
    cfg = load_yaml(str(config_path))
    
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    data_cfg = cfg["data"]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # 2. Data
    data_path = ROOT / data_cfg["train_path"]
    if not data_path.exists():
        logger.error(f"Dataset not found: {data_path}")
        # return or continue for dry run
    
    # Assuming same dataset class can handle binary labels if preprocessed correctly
    # or mapping happens inside dataset. For now assume dataset has 0/1 labels.
    train_ds = MultimodalDataset(str(data_path), split="train", split_ratio=0.8)
    val_ds = MultimodalDataset(str(data_path), split="val", split_ratio=0.8)
    
    train_loader = DataLoader(train_ds, batch_size=train_cfg["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=train_cfg["batch_size"], shuffle=False)

    # 3. Model
    # Initialize with PRETRAIN config logic essentially, then load weights
    # But here we construct with new num_classes immediately if we want to replace head?
    # Better strategy: Load pretrain model, then replace head.
    
    # Load Pretrain Config just to init structure correctly
    pretrain_ckpt = ROOT / "offline_training/artifacts/pretrain/best_checkpoint.pt"
    
    # Init with 4 classes first (matching pretrain)
    model = MultimodalClassifier(
        video_dim=model_cfg["video_dim"],
        audio_dim=model_cfg["audio_dim"],
        text_dim=model_cfg["text_dim"],
        meta_dim=model_cfg["meta_dim"],
        fusion_dim=model_cfg["fusion_dim"],
        num_classes=4, # Original pretrain classes
        dropout=model_cfg["dropout"],
        fusion_heads=model_cfg.get("fusion_heads", 8),
        fusion_layers=model_cfg.get("fusion_layers", 2)
    )
    
    if pretrain_ckpt.exists():
        logger.info(f"Loading pretrain weights from {pretrain_ckpt}")
        try:
            state_dict = torch.load(pretrain_ckpt, map_location=device)
            model.load_state_dict(state_dict)
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}")
    else:
        logger.warning("Pretrain checkpoint not found. Training from scratch provided config.")

    # Replace Head for 2 classes
    num_ft_classes = int(model_cfg["num_classes"]) # 2
    fusion_dim = model_cfg["fusion_dim"]
    
    model.classifier_head = nn.Sequential(
        nn.Dropout(model_cfg["dropout"]),
        nn.Linear(fusion_dim, fusion_dim // 2),
        nn.GELU(),
        nn.Linear(fusion_dim // 2, num_ft_classes)
    )
    model.to(device)

    # 4. Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=float(train_cfg["lr"]), weight_decay=float(train_cfg["weight_decay"]))
    
    epochs = int(train_cfg["epochs"])

    # 5. Loop
    best_f1 = 0.0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [FT]")
        for v, a, t, m, labels in pbar:
            v, a, t, m, labels = v.to(device), a.to(device), t.to(device), m.to(device), labels.to(device)
            
            # Remap labels if necessary?
            # Assuming dataset already has 0/1 for this task.
            # If dataset mixes 4 classes, we might need mapping here. 
            # For simplicity, assuming Phase 4 produced a binary dataset for Finetune or filtered it.
            
            optimizer.zero_grad()
            logits = model(v, a, t, m)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
        # Validation
        model.eval()
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for v, a, t, m, labels in val_loader:
                v, a, t, m, labels = v.to(device), a.to(device), t.to(device), m.to(device), labels.to(device)
                logits = model(v, a, t, m)
                preds = torch.argmax(logits, dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        val_acc = accuracy_score(val_labels, val_preds)
        val_f1 = f1_score(val_labels, val_preds, average="macro")
        
        # Specific Recall for "Harmful" (assuming class 1)
        # Check unique labels first
        recall_1 = recall_score(val_labels, val_preds, pos_label=1, zero_division=0) if num_ft_classes==2 else 0.0

        logger.info(f"Epoch {epoch+1}: Acc={val_acc:.4f} F1={val_f1:.4f} Recall(1)={recall_1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            save_path = ROOT / "offline_training/artifacts/finetune"
            save_path.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), save_path / "finetuned_best.pt")
            
            # ✨ Push to HF Hub
            try:
                commit_info = model.push_to_hub("funa21/tiktok-vn-finetune", commit_message=f"Epoch {epoch+1} F1={val_f1:.4f}")
                logger.info(f"Pushed to HF Hub: {commit_info.commit_url}")
            except Exception as e:
                logger.warning(f"Failed to push to HF Hub: {e}")

if __name__ == "__main__":
    train()
