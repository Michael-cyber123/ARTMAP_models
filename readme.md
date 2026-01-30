# Models

**eNeuralFAM** is a hybrid online classification framework that combines shallow Artificial Neural Networks (ANNs) for feature extraction with Adaptive Resonance Theory (ART)-based classifiers for incremental learning on streaming data.

**eNeuralFusAM** is a hybrid online classification framewor that combines shallow Artifical Neural Networks (ANNs) for feature extraction with Fusion ARTMAP (FusAM)-based classifiers for incremental learning on streaming data.

## Description

This repository implements and evaluates two ART-based online learning paradigms:

### 1) Concatenated (single-stream) model
- **ANN + Fuzzy ARTMAP (FAM)**: a single ANN extracts a low-dimensional embedding which is then used by FAM for online learning.

### 2) Modular (multi-channel) model
- **ANN + Fusion ARTMAP (FusAM)**: the input feature vector is split into multiple channels.  
A separate ANN is trained per channel to produce embeddings, which are then learned and fused by a multi-channel ARTMAP classifier.

This project implements an ensemble-based online learning model using:
- ANN for feature extraction
- Fuzzy ARTMAP (FAM) and Fusion ARTMAP (FusAM) as the classifier
- Ensemble voting mechanism for robust predictions

The model is evaluated across multiple datasets and noise levels, with performance tracked using various classification metrics.

## Datasets

The datasets used in this project can be downloaded from the [KEEL repository](http://keel.es/). 

## Evaluation Protocol

The dataset is stratified/shuffled and split into 10 folds:
- **Fold-0**: initialization (fit scalers, train ANN(s), train ART-based classifier)
- **Folds 1–9**: online phase (predict on fold → record metrics → update model on the same fold)

## Requirements

- Python 3.8+
- NumPy
- Pandas
- Scikit-learn
- TensorFlow
- scipy
- openpyxl
