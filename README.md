# Enhancing Two-Step Textual Anomaly Detection through Anisotropy Mitigation

This repository contains the code for the paper:

**"Enhancing Two-Step Textual Anomaly Detection through Anisotropy Mitigation"**  
Pierre Fihey, Matthieu Labeau, Pavlo Mozharovskyi

---

## Overview

Anomaly detection aims to distinguish between **in-distribution (ID)** samples and **out-of-distribution (OOD)** samples.

In NLP, recent approaches commonly follow a **two-stage framework**:
1. Extract embeddings from a pre-trained language model
2. Apply a classical anomaly detection algorithm on these embeddings

However, the **geometric structure of embedding spaces**—notably their anisotropy—can significantly impact the performance of detection algorithms.

### Key contributions

- We adapt diverse classification datasets from the MTEB benchmark for anomaly detection, enabling **large-scale multilingual and multi-domain evaluation**.
- We demonstrate the comparative advantage of **similarity-trained embedding models** for anomaly detection, linking it to geometric properties beneficial to detection algorithms.
- We demonstrate that a simple **post-processing step** can seemingly adapt embeddings to be used with most detection algorithms, greatly smoothing their variance in performance.

---

## 📊 Experiments

We evaluate our approach on a reformulation of **MTEB classification tasks** into anomaly detection problems.

- Multiple embedding models (XLM-RoBERTa-base, E5, Qwen3-Embeddings, Qwen3, LLaMA)
- Several anomaly detection methods:
  - KNN
  - LOF
  - Isolation Forest
  - One-Class SVM
  - GMM
  - LUNAR
- Various post-processing strategies:
  - Standard whitening
  - PCA whitening
  - Soft whitening
  - Flow-based transformations

---
