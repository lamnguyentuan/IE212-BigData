# Offline Training Module

This is the core Machine Learning component of the project. It handles everything from feature extraction to model training and evaluation.

## Workflow

1.  **Preprocessing**: Convert raw Video/Audio/Metadata into numerical feature vectors.
2.  **Pretraining**: Train the multimodal model on the TikHarm dataset (4 classes).
3.  **Finetuning**: Adapt the model to the specific Vietnamese TikTok context (Safe/Not Safe).

## Directory Layers

- `configs/`: YAML configuration files.
- `datasets/`: PyTorch Dataset implementations (`MultimodalDataset`).
- `features/`: Feature builder logic.
- `models/`: PyTorch model architecture definitions.
- `preprocessing/`: Feature extraction scripts (TimeSformer, Wav2Vec2, BERT).
- `pretrain/`: Pretraining scripts.
- `finetune/`: Finetuning scripts.
