---
title: polyinverse
app_file: demo_v2.py
sdk: gradio
sdk_version: 6.13.0
---
# 🧪 Polyinverse
## AI-powered Polymer Inverse Design

[![Hugging Face Spaces](https://img.shields.io/badge/🤗%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/00mahoon/polyinverse)
[![GitHub](https://img.shields.io/badge/GitHub-polyinverse-black)](https://github.com/00mahoon/polyinverse)

> **"Tell the AI what properties you want. It designs the molecule."**

---

## 🎯 What is Polyinverse?

Polyinverse is an AI system that predicts polymer properties from molecular structure (SMILES) — and reverses the process to **design new polymer candidates** from target properties.

Inspired by AlphaFold's approach to protein structure prediction, Polyinverse applies Graph Neural Networks to the challenge of polymer inverse design.

---

## ✨ Features

### 🔬 Forward Prediction
- Input: Polymer SMILES string
- Output: Predicted **Density** and **Tc** (crystallization temperature)
- Model: Multi-task Graph Neural Network
- Performance: R² = 0.757 (Density), R² = 0.512 (Tc)

### 🎯 Inverse Design
- Input: Target density value
- Output: Top candidate polymer structures
- Method: Genetic algorithm with GNN-based fitness function
- Achieves error < 0.01 g/ml in most cases

---

## 🚀 Try it now

**Live Demo:** [https://huggingface.co/spaces/00mahoon/polyinverse](https://huggingface.co/spaces/00mahoon/polyinverse)

---

## 📊 Model Performance

| Property | R² Score | MAE |
|----------|----------|-----|
| Density  | 0.757    | 0.0294 g/ml |
| Tc       | 0.512    | 0.0364 |

Trained on **NeurIPS 2025 Open Polymer Prediction Challenge** dataset (3,502 samples).

---

## 🛠 Tech Stack

| Component | Technology |
|-----------|------------|
| GNN Model | PyTorch Geometric (GCNConv) |
| Molecular Features | RDKit |
| Inverse Design | Genetic Algorithm |
| Web Interface | Gradio |
| Deployment | Hugging Face Spaces |

---

## 📁 Project Structure
---

## 🗺 Roadmap

- [x] Data exploration & baseline models
- [x] Graph Neural Network (R² 0.757)
- [x] Inverse design with genetic algorithm
- [x] Hugging Face Spaces deployment
- [ ] More property predictions (Tg, FFV, Rg)
- [ ] Molecular generation with VAE/Diffusion


---

## 💡 Motivation

Traditional polymer development relies on trial-and-error experimentation.
Polyinverse flips this process:
---

## 👨‍🔬 About

Built by a materials engineer with 9 years at Henkel + CS background.
The intersection of domain expertise and AI — making materials discovery faster.

**Goal:** Accelerate sustainable materials development through AI.

---

## 📬 Contact

- GitHub: [@00mahoon](https://github.com/00mahoon)
- Hugging Face: [@00mahoon](https://huggingface.co/00mahoon)

---

*"The best way to predict the future is to invent it." — Alan Kay*
