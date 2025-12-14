
import sys
import torch
import torch.nn as nn
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from common.models.multimodal_classifier import MultimodalClassifier
from common.utils.logging_utils import get_logger

logger = get_logger("eval-real-tiktok")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_real_data(test_size=0.3):
    data_path = ROOT / "evaluation/dataset_tiktok_real_verified.npz"
    if not data_path.exists():
        raise FileNotFoundError(f"{data_path} not found. Run prepare_tiktok_real_dataset.py first.")
        
    data = np.load(data_path, allow_pickle=True)
    
    # Features
    video_embs = torch.tensor(data['video_embs'], dtype=torch.float32)
    audio_embs = torch.tensor(data['audio_embs'], dtype=torch.float32)
    text_embs = torch.tensor(data['text_embs'], dtype=torch.float32)
    meta_nums = torch.tensor(data['metadata_numeric'], dtype=torch.float32)
    labels = torch.tensor(data['labels'], dtype=torch.long)
    
    # Ensure Binary Labels for standard eval (0=Safe, >0=Harmful)
    # The manual labels are 0/1, but if any stray 2/3 exist, map them.
    labels = (labels > 0).long()
    
    # REPLICATE TRAINING SPLIT LOGIC STRICTLY
    # offline_training/datasets/multimodal_dataset.py matches this:
    # np.random.seed(42)
    # np.random.shuffle(indices)
    # val = indices[int(0.8 * len):]
    
    indices = np.arange(len(labels))
    np.random.seed(42)
    np.random.shuffle(indices)
    
    split_ratio = 0.8
    split_point = int(len(indices) * split_ratio)
    
    # We evaluate on the VALIDATION set (held out during training updates)
    test_idx = indices[split_point:]
    
    logger.info(f"Total Samples: {len(labels)}")
    logger.info(f"Using Validation Split (Last 20%): {len(test_idx)} samples")
    
    test_ds = TensorDataset(
        video_embs[test_idx], 
        audio_embs[test_idx], 
        text_embs[test_idx], 
        meta_nums[test_idx], 
        labels[test_idx]
    )
    
    return DataLoader(test_ds, batch_size=32, shuffle=False)

def load_model(model_type, num_classes):
    # Determine path
    if model_type == "pretrain":
        local_ckpt = ROOT / "offline_training/artifacts/pretrain/best_checkpoint.pt"
        hf_repo = "funa21/tiktok-harm-clip-v2"
    else: # finetune
        local_ckpt = ROOT / "offline_training/artifacts/finetune/finetuned_best.pt"
        hf_repo = "funa21/tiktok-vn-finetune"
        
    ckpt_path = None
    if local_ckpt.exists():
        logger.info(f"Loading local {model_type} from {local_ckpt}")
        ckpt_path = str(local_ckpt)
    else:
        logger.info(f"Local {model_type} missing, downloading from HF: {hf_repo}")
        try:
            from huggingface_hub import hf_hub_download
            ckpt_path = hf_hub_download(repo_id=hf_repo, filename="model.safetensors")
        except:
             try:
                ckpt_path = hf_hub_download(repo_id=hf_repo, filename="pytorch_model.bin")
             except Exception as e:
                logger.error(f"Failed to download {model_type}: {e}")
                raise

    # Both models expect 22 dims
    meta_dim = 22 
    
    model = MultimodalClassifier(
        video_dim=768,
        audio_dim=768,
        text_dim=768,
        meta_dim=meta_dim, 
        num_classes=num_classes,
        fusion_dim=768
    )
    
    # Load weights
    try:
        from safetensors.torch import load_file
        state_dict = load_file(ckpt_path)
    except:
        state_dict = torch.load(ckpt_path, map_location=device)
        
    model.load_state_dict(state_dict, strict=False) # Strict=False to avoid minor mismatches if any
    model.to(device)
    model.eval()
    return model

def evaluate_model(model, loader, model_name, is_pretrain=False):
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for v, a, t, m, labels in loader:
            v, a, t, m = v.to(device), a.to(device), t.to(device), m.to(device)
            
            # Both Pretrain AND Finetune seem to expect 22 dims (inherited from TikHarm)
            # We have 6. Pad with zeros.
            if m.shape[1] < 22:
                padding = torch.zeros(m.shape[0], 22 - m.shape[1], device=device)
                m = torch.cat([m, padding], dim=1)
                
            logits = model(v, a, t, m)
            probs = torch.softmax(logits, dim=1)
            
            if is_pretrain:
                # 4 Classes: 0=Safe, 1,2,3=Harmful
                # Prob of harmful is sum of probs for 1,2,3 (or 1 - prob[0])
                prob_harmful = 1.0 - probs[:, 0]
                pred_binary = (torch.argmax(probs, dim=1) != 0).long()
            else:
                # 2 Classes: 0=Safe, 1=Harmful
                prob_harmful = probs[:, 1]
                pred_binary = torch.argmax(probs, dim=1)
                
            all_preds.extend(pred_binary.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(prob_harmful.cpu().numpy())
            
    # Metrics
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except:
        auc = 0.0
        
    conf_mat = confusion_matrix(all_labels, all_preds)
    
    print(f"\n--- Results for {model_name} ---")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUC: {auc:.4f}")
    print(f"Confusion Matrix (TN, FP, FN, TP):\n{conf_mat}")
    
    # Save results for visualization
    results_dir = ROOT / "evaluation/results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    filename = "pretrain_results.npz" if is_pretrain else "finetune_results.npz"
    save_path = results_dir / filename
    
    np.savez_compressed(
        save_path,
        preds=np.array(all_preds),
        labels=np.array(all_labels),
        probs=np.array(all_probs),
        model_name=model_name
    )
    logger.info(f"Saved results to {save_path}")
    
    return {
        "acc": acc, "prec": prec, "rec": rec, "f1": f1, "auc": auc
    }

def main():
    print("Loading Data (30% Test Split)...")
    test_loader = load_real_data(test_size=0.3)
    
    # 1. Evaluate Pretrain (Baseline)
    print("\nLoading Pretrain Model (4 classes)...")
    model_pre = load_model("pretrain", num_classes=4)
    res_pre = evaluate_model(model_pre, test_loader, "Pretrain (Baseline)", is_pretrain=True)
    
    # 2. Evaluate Finetune
    print("\nLoading Finetune Model (2 classes)...")
    model_fine = load_model("finetune", num_classes=2)
    res_fine = evaluate_model(model_fine, test_loader, "Finetune (TikTok-VN)", is_pretrain=False)
    
    # Comparison
    print("\n--- Summary Comparison ---")
    print(f"{'Metric':<10} | {'Pretrain':<10} | {'Finetune':<10} | {'Diff':<10}")
    print("-" * 46)
    for m in ["acc", "prec", "rec", "f1", "auc"]:
        v1 = res_pre[m]
        v2 = res_fine[m]
        diff = v2 - v1
        print(f"{m.upper():<10} | {v1:.4f}     | {v2:.4f}     | {diff:+.4f}")

if __name__ == "__main__":
    main()
