# 🧠 xLSTM: Extended Long Short-Term Memory Implementation

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Gradio](https://img.shields.io/badge/Gradio-726E88?style=for-the-badge&logo=gradio&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)

A PyTorch implementation and benchmarking study of the **xLSTM (Extended LSTM)** architecture. This project explores how exponential gating and matrix memory structures allow Recurrent Neural Networks to overcome traditional LSTM limitations and achieve Transformer-like performance with linear computation complexity.

## 📖 Overview

This repository implements the key components of the xLSTM architecture:
1.  **sLSTM (Scalar LSTM):** Uses exponential gating and memory mixing for better gradient flow.
2.  **mLSTM (Matrix LSTM):** Utilizes a matrix memory structure (Key-Value pairs) to enhance storage capacity, acting as a bridge between RNNs and Transformers.

The project benchmarks these architectures against a standard Vanilla LSTM on character-level language modeling (Tiny Shakespeare).

## 🏗️ Architectures Implemented

### 1. sLSTM (Scalar LSTM)
The sLSTM modifies the traditional LSTM by replacing the sigmoid input gate with an exponential gate. This prevents the vanishing gradient problem and allows the model to revise storage decisions more effectively.

* **Key Feature:** Exponential Gating (`torch.exp`)
* **Normalization:** Stabilizer state ($n_t$) tracks input magnitude.

### 2. mLSTM (Matrix LSTM) The mLSTM replaces the scalar cell state with a matrix, significantly increasing memory capacity. It operates similarly to the Self-Attention mechanism in Transformers but retains the recurrent $O(N)$ inference complexity.

* **Key-Value Memory:** Updates memory via $v \otimes k^T$ (outer product).
* **Retrieval:** Query vector $q$ retrieves data via matrix multiplication.

## 📊 Benchmarks & Results

We trained both a **Vanilla LSTM** and the **xLSTM (sLSTM)** on the Tiny Shakespeare dataset with identical hyperparameters (Embedding Dim: 128, Hidden Size: 256).

| Metric | Vanilla LSTM | xLSTM (Ours) |
| :--- | :--- | :--- |
| **Convergence Speed** | Slower | **Faster** |
| **Final Loss (1k steps)** | ~1.81 | **~1.66** |
| **Gradient Flow** | Standard | Improved (via Exp Gating) |

*The xLSTM demonstrates superior convergence stability and lower cross-entropy loss compared to the baseline.*

## 🚀 Getting Started

### Prerequisites
* Python 3.8+
* PyTorch
* Gradio (for the demo)
* Matplotlib

### Installation
```bash
git clone [https://github.com/yourusername/xLSTM-PyTorch-Bench.git](https://github.com/yourusername/xLSTM-PyTorch-Bench.git)
cd xLSTM-PyTorch-Bench
pip install torch gradio matplotlib requests
