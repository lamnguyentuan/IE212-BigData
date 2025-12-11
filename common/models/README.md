# Multimodal Model Architecture

We utilize an **Early Fusion** architecture combining three modalities.

## Components

### 1. Projectors (`projections.py`)
Each modality (Video, Audio, Text, Metadata) is projected into a shared latent space (`feature_dim=768`) via linear layers and normalization.

### 2. Fusion Module (`fusion.py`)
A **Cross-Attention** mechanism is used to fuse the projected embeddings.
- **Query**: Text + Metadata features
- **Key/Value**: Video + Audio features
- **Architecture**: Stacked Multihead Attention layers followed by pooling.
- **Output**: `[Batch, fusion_dim]` pooled vector ready for classification.

### 3. Classifier Head (`multimodal_classifier.py`)
A Multi-Layer Perceptron (MLP) mapping the fused vector to class probabilities.
- **Pretraining**: 4 outputs (Adult, Harmful, Safe, Suicide).
- **Finetuning**: 2 outputs (Safe, Not Safe).

## Integration
The model now supports **Hugging Face Hub** integration using `PyTorchModelHubMixin`, allowing seamless `push_to_hub()` and `from_pretrained()`.
