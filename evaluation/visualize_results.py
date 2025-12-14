
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import pandas as pd

# Define paths
ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = Path("evaluation/results")
SAVE_DIR = Path("evaluation")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

def load_results():
    files = {
        "Baseline (Pretrain)": RESULTS_DIR / "pretrain_results.npz",
        "Finetune (TikTok-VN)": RESULTS_DIR / "finetune_results.npz"
    }
    
    results = {}
    for name, path in files.items():
        full_path = ROOT / path
        if full_path.exists():
            data = np.load(full_path, allow_pickle=True)
            results[name] = {
                "preds": data["preds"],
                "labels": data["labels"],
                "probs": data["probs"],
            }
        else:
            print(f"Warning: {full_path} not found.")
    return results

def plot_metrics_comparison(results):
    if not results: return
    
    metrics_data = [] # List of dicts
    
    for model_name, res in results.items():
        labels = res["labels"]
        preds = res["preds"]
        
        acc = accuracy_score(labels, preds)
        prec = precision_score(labels, preds, zero_division=0)
        rec = recall_score(labels, preds, zero_division=0)
        f1 = f1_score(labels, preds, zero_division=0)
        
        metrics_data.append({"Model": model_name, "Metric": "Accuracy", "Value": acc})
        metrics_data.append({"Model": model_name, "Metric": "Precision (Harmful)", "Value": prec})
        metrics_data.append({"Model": model_name, "Metric": "Recall (Harmful)", "Value": rec})
        metrics_data.append({"Model": model_name, "Metric": "F1 Score", "Value": f1})
        
    df_metrics = pd.DataFrame(metrics_data)
    
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    sns.barplot(data=df_metrics, x="Metric", y="Value", hue="Model", palette="viridis")
    
    plt.title("Model Performance Comparison on Verified TikTok Data (553 Videos)", fontsize=14)
    plt.ylim(0, 1.05)
    plt.legend(loc='lower left')
    
    # Add values on bars
    for container in plt.gca().containers:
        plt.gca().bar_label(container, fmt='%.2f', padding=3)

    plt.tight_layout()
    save_path = SAVE_DIR / "comparison_metrics_bar.png"
    plt.savefig(save_path)
    print(f"Saved {save_path}")
    plt.close()

def plot_confusion_matrices(results):
    if not results: return
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    for ax, (model_name, res) in zip(axes, results.items()):
        cm = confusion_matrix(res["labels"], res["preds"])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Safe', 'Harmful'],
                    yticklabels=['Safe', 'Harmful'])
        ax.set_title(f"{model_name}")
        ax.set_ylabel('Actual')
        ax.set_xlabel('Predicted')
        
    plt.suptitle("Confusion Matrices (Verified Validation Set)", fontsize=14)
    plt.tight_layout()
    save_path = SAVE_DIR / "comparison_confusion_matrices.png"
    plt.savefig(save_path)
    print(f"Saved {save_path}")
    plt.close()

def plot_roc_curves(results):
    if not results: return
    
    plt.figure(figsize=(8, 6))
    
    for model_name, res in results.items():
        fpr, tpr, _ = roc_curve(res["labels"], res["probs"])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.2f})', linewidth=2)
        
    plt.plot([0, 1], [0, 1], 'k--', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves (Verified Validation Set)')
    plt.legend(loc="lower right")
    
    save_path = SAVE_DIR / "comparison_roc_curve.png"
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved {save_path}")
    plt.close()

def main():
    print("Loading results...")
    results = load_results()
    if not results:
        print("No data found. Run eval_tiktok_real.py first.")
        return
        
    print("Generating visualizations...")
    plot_metrics_comparison(results)
    plot_confusion_matrices(results)
    plot_roc_curves(results)
    print("Done.")

if __name__ == "__main__":
    main()
