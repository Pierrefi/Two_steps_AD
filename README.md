# Enhancing Two-Step Textual Anomaly Detection through Anisotropy Mitigation

This repository contains the code for the paper:

**"Enhancing Two-Step Textual Anomaly Detection through Anisotropy Mitigation"**  
Pierre Fihey, Matthieu Labeau, Pavlo Mozharovskyi

---

## 🧠 Overview

Anomaly detection aims to distinguish between **in-distribution (ID)** samples and **out-of-distribution (OOD)** samples.

In NLP, recent approaches commonly follow a **two-stage framework**:
1. Extract embeddings from a pre-trained language model
2. Apply a classical anomaly detection algorithm on these embeddings

However, the **geometric structure of embedding spaces**—notably their anisotropy—can significantly impact the performance of detection algorithms.

### Key contributions

- We highlight the importance of **similarity-trained embedding models** for anomaly detection
- We show that their **geometric properties** are better aligned with distance-based detectors
- We demonstrate that a simple **post-processing step (whitening)** can:
  - reduce anisotropy
  - significantly improve performance
  - homogenize results across detection algorithms

---

## 📊 Experiments

We evaluate our approach on a reformulation of **MTEB classification tasks** into anomaly detection problems.

- Multiple embedding models (e.g., E5, Qwen3, LLaMA)
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
