"""
Phase 2 — ECG Upsampling Pre-training

Self-supervised pre-training combining:
  - MAE (Masked Autoencoder) with rhythm-aware masking
  - NT-Xent contrastive learning with ECG augmentations

The HybridEncoder (CNN + Transformer) trained here is exported for Phase 3.
"""
