# reliability-ai

Time-series anomaly detection for industrial equipment.

## Purpose

`reliability-ai` provides development tools to deploy machine learning models for anomaly detection across systems.

This implementation focuses on Temporal Convolutional Network Autoencoders (TCN-AE) for multivariate time-series anomaly detection.

## Application

The first implementation is based on high-pressure compressor anomaly detection.

The library is designed such that system-specific information:

- feature names
- window sizes
- model parameters
- anomaly thresholds

can be configured independently from model implementation.

## Installation

For development, install the package in editable mode:

```bash
pip install -e .