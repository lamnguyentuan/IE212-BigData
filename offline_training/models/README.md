# Multimodal Model Architecture

We utilize an **Early Fusion** architecture combining three modalities.

## Components

### 1. Projectors (`projections.py`)
Each modality (Video, Audio, Text, Metadata) is projected into a shared latent space (`feature_dim=768`) via linear layers and normalization.

### 2. Fusion Module (`fusion.py`)
A **Self-Attention** mechanism (Transformer Encoder Layer) is used to fuse the projected embeddings.
- Input: `[Batch, 4, 768]` (Sequence of 4 modalities)
- Output: `[Batch, 4, 768]` -> Pooled to `[Batch, 768]`

### 3. Classifier Head (`multimodal_classifier.py`)
A Multi-Layer Perceptron (MLP) mapping the fused vector to class probabilities.
- **Pretraining**: 4 outputs (Adult, Harmful, Safe, Suicide).
- **Finetuning**: 2 outputs (Safe, Not Safe).

## Integration
The model now supports **Hugging Face Hub** integration using `PyTorchModelHubMixin`, allowing seamless `push_to_hub()` and `from_pretrained()`.
