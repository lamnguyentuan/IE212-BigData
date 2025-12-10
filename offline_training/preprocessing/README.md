# Multimodal Preprocessing

This pipeline transforms raw media files into rich feature embeddings stored in the **Silver Layer**.

## Feature Extractors

### 1. Video Features (`TimeSformer`)
- **Input**: `.mp4` video.
- **Process**: Sample 8 frames -> Resize (224x224) -> TimeSformer Model.
- **Output**: 768-dim embedding (`video_embedding.npy`).

### 2. Audio Features (`Wav2Vec2`)
- **Input**: Extracted `.wav` audio.
- **Process**: Resample 16kHz -> Wav2Vec2 Base.
- **Output**: 768-dim embedding (`audio_embedding.npy`).

### 3. Metadata/Text Features (`PhoBERT` + Statistical)
- **Input**: Description, numeric stats (likes, shares, duration).
- **Process**: 
    - Text: Tokenize -> PhoBERT -> 768-dim embedding.
    - Numeric: Normalize/Scale -> Concatenate.
- **Output**: `.npz` file containing text and metadata vectors.

## Running the Pipeline

Execute the full pipeline driver:
```bash
python pipelines/preprocess_full_pipeline.py
```
This script orchestrates the extraction for all unprocessed videos in MinIO.
