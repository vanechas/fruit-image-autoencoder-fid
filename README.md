# 🍎 Fruit Image Reconstruction Using Convolutional Autoencoders & FID Benchmarking

A Computer Vision deep learning project focused on self-supervised image reconstruction across multiple fruit quality classes using deep Convolutional Autoencoders (CAE)[cite: 9]. Model generative fidelity is quantitatively evaluated using Fréchet Inception Distance (FID) extracted via InceptionV3[cite: 9].

---

## 📌 Project Overview

* **Domain:** Computer Vision, Generative Modeling, Unsupervised Feature Learning[cite: 9]
* **Task:** Image reconstruction and latent space compression for fresh and rotten fruit classes[cite: 9]
* **Dataset:** 6 classes (`freshapples`, `freshbanana`, `freshoranges`, `rottenapples`, `rottenbanana`, `rottenoranges`)[cite: 9]:
  * Training set: 2,778 images[cite: 9]
  * Validation set: 2,353 images[cite: 9]
  * Resolution: $100 \times 100 \times 3$[cite: 9]

---

## 🏗️ Model Architectures

### 1. Baseline Autoencoder
* **Encoder:** Sequential `Conv2D` layers (32, 64, 128 filters) with `ReLU` activations and `MaxPooling2D` downsampling[cite: 9].
* **Decoder:** Symmetrical `UpSampling2D` and `Conv2D` blocks with final sigmoid activation[cite: 9].
* **Loss:** Mean Squared Error (MSE)[cite: 9].

### 2. Improved Autoencoder
* Upgraded with **Batch Normalization** and **LeakyReLU (slope 0.2)** to prevent dead neurons and stabilize gradient flow[cite: 9].
* Latent representation expanded to 256 channels at bottleneck ($25 \times 25 \times 256$)[cite: 9].
* Optimized with Adam optimizer ($lr = 10^{-4}$) over a prebatched `tf.data` input pipeline[cite: 9].

---

## 📊 Evaluation & FID Results

Reconstruction fidelity on unseen test images was measured against the real data distribution using **Fréchet Inception Distance (FID)** with an InceptionV3 backbone[cite: 9]:

| Fruit Class | FID Score (Lower is better) | Quality Assessment |
| :--- | :---: | :--- |
| `freshbanana` | **1.24** | Highest reconstruction fidelity[cite: 9] |
| `rottenoranges` | **1.50** | Excellent structural preservation[cite: 9] |
| `rottenbanana` | **1.69** | Sharp edge and texture recovery[cite: 9] |
| `rottenapples` | **1.83** | Accurate blemish reconstruction[cite: 9] |
| `freshapples` | **2.03** | Smooth gradient and surface capture[cite: 9] |
| `freshoranges` | **2.45** | Moderate surface texture variance[cite: 9] |

---

## 🛠️ Tech Stack & Requirements

* **Framework:** TensorFlow 2.x, Keras[cite: 9]
* **Scientific Computing:** NumPy, SciPy (`sqrtm`), scikit-learn[cite: 9]
* **Feature Extractor:** `InceptionV3` (`tf.keras.applications`)[cite: 9]
* **Visualization:** Matplotlib[cite: 9]
