# reliability-ai

A modular Python package for multivariate time-series anomaly detection using a TensorFlow/Keras Temporal Convolutional Network Autoencoder.

## Current capabilities

- Causal dilated Conv1D residual blocks
- TCN autoencoder reconstruction
- Contiguous time-series sequence generation
- StandardScaler preprocessing
- Training and validation with early stopping
- Sequence-level reconstruction error
- Feature-level reconstruction attribution
- Percentile-based anomaly detection
- Timestamp-aligned engineering-unit reconstruction results
- Configuration-driven high-level detector
- Automated pytest coverage

## Status

Version 0.1.0 is under active development.

A reproducible demonstration using the NASA C-MAPSS turbofan degradation dataset is planned for a later release.

## Installation

```bash
git clone https://github.com/darylchialerchoo/reliability-ai.git
cd reliability-ai
python -m venv venv
pip install -e ".[dev]"
