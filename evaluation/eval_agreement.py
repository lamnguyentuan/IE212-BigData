import json
import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix, accuracy_score

FILE_1 = "offline_training/preprocessing/tiktok_label_done.json"
FILE_2 = "offline_training/preprocessing/tiktok_label_second.json"

def load_labels(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    labels = {}
    for task in data:
        # Use video filename or S3 path as unique key
        video_path = task['data']['video']
        
        # Get label from first annotation
        if not task['annotations']:
            continue
            
        try:
            result = task['annotations'][0]['result']
            if not result:
                continue
            
            label = result[0]['value']['choices'][0]
            labels[video_path] = label
        except (KeyError, IndexError):
            continue
            
    return labels

def main():
    print(f"Loading Annotator 1: {FILE_1}")
    labels_1 = load_labels(FILE_1)
    
    print(f"Loading Annotator 2: {FILE_2}")
    labels_2 = load_labels(FILE_2)
    
    # Find common keys
    common_keys = set(labels_1.keys()) & set(labels_2.keys())
    print(f"Found {len(common_keys)} common videos labeled by both.")
    
    if len(common_keys) == 0:
        print("No overlapping data found. Please check if the datasets match.")
        return

    y1 = []
    y2 = []
    
    for k in common_keys:
        y1.append(labels_1[k])
        y2.append(labels_2[k])
        
    # Metrics
    acc = accuracy_score(y1, y2)
    kappa = cohen_kappa_score(y1, y2)
    conf_mat = confusion_matrix(y1, y2, labels=["SAFE", "NOT SAFE"])
    
    print("\n--- Agreement Results ---")
    print(f"Total Overlap: {len(common_keys)}")
    print(f"Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print(f"Cohen's Kappa: {kappa:.4f}")
    
    print("\n--- Confusion Matrix ---")
    print("                [Annotator 2]")
    print("                SAFE   NOT SAFE")
    print(f"[Annotator 1] SAFE      {conf_mat[0][0]:<6} {conf_mat[0][1]:<6}")
    print(f"              NOT SAFE  {conf_mat[1][0]:<6} {conf_mat[1][1]:<6}")
    
    print("\n--- Interpretation ---")
    if kappa < 0:
        print("Kappa < 0: Poor agreement (less than chance).")
    elif kappa <= 0.20:
        print("Kappa 0.01-0.20: Slight agreement.")
    elif kappa <= 0.40:
        print("Kappa 0.21-0.40: Fair agreement.")
    elif kappa <= 0.60:
        print("Kappa 0.41-0.60: Moderate agreement.")
    elif kappa <= 0.80:
        print("Kappa 0.61-0.80: Substantial agreement.")
    else:
        print("Kappa 0.81-1.00: Almost perfect agreement.")

if __name__ == "__main__":
    main()
