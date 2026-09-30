# 🍎 Fruit Image Reconstruction Using Convolutional Autoencoders & FID Benchmarking

A Computer Vision deep learning project focused on self-supervised image reconstruction across multiple fruit quality classes using deep Convolutional Autoencoders (CAE). Model generative fidelity is quantitatively evaluated using Fréchet Inception Distance (FID) extracted via InceptionV3.

---

## 📌 Project Overview

* **Domain:** Computer Vision, Generative Modeling, Unsupervised Feature Learning
* **Task:** Image reconstruction and latent space compression for fresh and rotten fruit classes
* **Dataset:** 6 classes (`freshapples`, `freshbanana`, `freshoranges`, `rottenapples`, `rottenbanana`, `rottenoranges`):
  * Training set: 2,778 images
  * Validation set: 2,353 images
  * Resolution: $100 \times 100 \times 3$

---

## 🏗️ Model Architectures

### 1. Baseline Autoencoder
* **Encoder:** Sequential `Conv2D` layers (32, 64, 128 filters) with `ReLU` activations and `MaxPooling2D` downsampling.
* **Decoder:** Symmetrical `UpSampling2D` and `Conv2D` blocks with final sigmoid activation.
* **Loss:** Mean Squared Error (MSE).

### 2. Improved Autoencoder
* Upgraded with **Batch Normalization** and **LeakyReLU (slope 0.2)** to prevent dead neurons and stabilize gradient flow.
* Latent representation expanded to 256 channels at bottleneck ($25 \times 25 \times 256$).
* Optimized with Adam optimizer ($lr = 10^{-4}$) over a prebatched `tf.data` input pipeline.

---

## 📊 Evaluation & FID Results

Reconstruction fidelity on unseen test images was measured against the real data distribution using **Fréchet Inception Distance (FID)** with an InceptionV3 backbone:

| Fruit Class | FID Score (Lower is better) | Quality Assessment |
| :--- | :---: | :--- |
| `freshbanana` | **1.24** | Highest reconstruction fidelity |
| `rottenoranges` | **1.50** | Excellent structural preservation |
| `rottenbanana` | **1.69** | Sharp edge and texture recovery |
| `rottenapples` | **1.83** | Accurate blemish reconstruction |
| `freshapples` | **2.03** | Smooth gradient and surface capture |
| `freshoranges` | **2.45** | Moderate surface texture variance |

---

## 🛠️ Tech Stack & Requirements

* **Framework:** TensorFlow 2.x, Keras
* **Scientific Computing:** NumPy, SciPy (`sqrtm`), scikit-learn
* **Feature Extractor:** `InceptionV3` (`tf.keras.applications`)
* **Visualization:** Matplotlib
