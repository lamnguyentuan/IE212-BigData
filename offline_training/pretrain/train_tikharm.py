"""
Pretraining Script for TikHarm (4-class classification).
"""

import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score

# Paths
ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT))

from offline_training.utils.config_loader import load_yaml
from offline_training.models.multimodal_classifier import MultimodalClassifier
from offline_training.datasets.multimodal_dataset import MultimodalDataset
from offline_training.preprocessing.utils.logging_utils import get_logger

logger = get_logger("pretrain-tikharm")

def train():
    # 1. Load Config
    config_path = Path(__file__).parent / "pretrain_config.yaml"
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
        return

    train_ds = MultimodalDataset(str(data_path), split="train", split_ratio=0.9)
    val_ds = MultimodalDataset(str(data_path), split="val", split_ratio=0.9)
    
    train_loader = DataLoader(train_ds, batch_size=train_cfg["batch_size"], shuffle=True, num_workers=train_cfg.get("num_workers", 2))
    val_loader = DataLoader(val_ds, batch_size=train_cfg["batch_size"], shuffle=False, num_workers=train_cfg.get("num_workers", 2))
    
    logger.info(f"Train size: {len(train_ds)}, Val size: {len(val_ds)}")

    # 3. Model
    model = MultimodalClassifier(
        video_dim=model_cfg["video_dim"],
        audio_dim=model_cfg["audio_dim"],
        text_dim=model_cfg["text_dim"],
        meta_dim=model_cfg["meta_dim"],
        fusion_dim=model_cfg["fusion_dim"],
        num_classes=model_cfg["num_classes"],
        dropout=model_cfg["dropout"],
        fusion_heads=model_cfg.get("fusion_heads", 8),
        fusion_layers=model_cfg.get("fusion_layers", 2)
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=float(train_cfg["lr"]), weight_decay=float(train_cfg["weight_decay"]))
    
    epochs = int(train_cfg["epochs"])
    
    # 4. Loop
    best_f1 = 0.0
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for v, a, t, m, labels in pbar:
            v, a, t, m, labels = v.to(device), a.to(device), t.to(device), m.to(device), labels.to(device)
            
            optimizer.zero_grad()
            logits = model(v, a, t, m)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
        train_acc = accuracy_score(all_labels, all_preds)
        train_loss = running_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for v, a, t, m, labels in val_loader:
                v, a, t, m, labels = v.to(device), a.to(device), t.to(device), m.to(device), labels.to(device)
                logits = model(v, a, t, m)
                loss = criterion(logits, labels)
                val_loss += loss.item()
                preds = torch.argmax(logits, dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        val_acc = accuracy_score(val_labels, val_preds)
        val_f1 = f1_score(val_labels, val_preds, average="macro")
        val_loss /= len(val_loader)
        
        logger.info(f"Epoch {epoch+1}: Train Loss={train_loss:.4f} Acc={train_acc:.4f} | Val Loss={val_loss:.4f} Acc={val_acc:.4f} F1={val_f1:.4f}")
        
        # Save Checkpoint
        if val_f1 > best_f1:
            best_f1 = val_f1
            save_path = ROOT / "offline_training/artifacts/pretrain"
            save_path.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), save_path / "best_checkpoint.pt")
            logger.info("Saved new best checkpoint.")
            
            # ✨ Push to HF Hub
            try:
                commit_info = model.push_to_hub("funa21/tikharm-multimodal-pretrain", commit_message=f"Epoch {epoch+1} F1={val_f1:.4f}")
                logger.info(f"Pushed to HF Hub: {commit_info.commit_url}")
            except Exception as e:
                logger.warning(f"Failed to push to HF Hub: {e}")

if __name__ == "__main__":
    train()
