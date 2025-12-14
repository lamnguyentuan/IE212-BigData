import json
from pathlib import Path

def analyze_labels():
    label_file = Path("/home/funalee/UIT/IE104/project/IE212-BigData/offline_training/preprocessing/tiktok_label_done.json")
    
    if not label_file.exists():
        print(f"File not found: {label_file}")
        return

    with open(label_file, 'r') as f:
        data = json.load(f)
        
    print(f"Total entries in label file: {len(data)}")
    
    safe_count = 0
    not_safe_count = 0
    unknown_count = 0
    
    for item in data:
        # Check annotations
        annotations = item.get('annotations', [])
        if not annotations:
            unknown_count += 1
            continue
            
        # Assume last annotation is valid one or check 'ground_truth'
        # Taking the first annotation's result for now as per sample
        try:
            result = annotations[0].get('result', [])
            if not result:
                unknown_count += 1
                continue
                
            value = result[0].get('value', {})
            choices = value.get('choices', [])
            
            if "SAFE" in choices:
                safe_count += 1
            elif "NOT SAFE" in choices:
                not_safe_count += 1
            else:
                unknown_count += 1
                print(f"Unknown choice: {choices}")
        except Exception as e:
            print(f"Error parsing item {item.get('id')}: {e}")
            unknown_count += 1
            
    print("\n--- TikTok Data Label Analysis (from tiktok_label_done.json) ---")
    print(f"Total Videos: {len(data)}")
    print(f"Safe: {safe_count}")
    print(f"Not Safe: {not_safe_count}")
    print(f"Unknown: {unknown_count}")

if __name__ == "__main__":
    analyze_labels()
