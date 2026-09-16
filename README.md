# PDCOVIDNet: Dual-Branch Dilated CNN for COVID-19 Detection from Chest X-Rays

A deep learning project that classifies chest X-ray images into **COVID-19**, **Normal**, and **Viral Pneumonia** categories using a custom convolutional neural network architecture with parallel dilated convolution branches.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Data Augmentation](#data-augmentation)
- [Training Setup](#training-setup)
- [Results](#results)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Key Insights](#key-insights)
- [Future Improvements](#future-improvements)
- [License](#license)

---

## 🩺 Overview

Chest X-ray imaging is a fast, low-cost, and widely available diagnostic tool that can assist in distinguishing COVID-19 pneumonia from other respiratory conditions. This project implements **PDCOVIDNet**, a custom convolutional neural network that processes X-ray images through two parallel branches with different receptive fields (standard and dilated convolutions), then fuses their features for final classification.

The notebook covers the complete deep learning pipeline:

1. Mounting and loading image data from Google Drive
2. Data augmentation to improve generalization
3. Designing a dual-branch dilated CNN architecture from scratch
4. Training with checkpointing and learning-rate scheduling
5. Evaluating performance on validation and test sets
6. Visualizing training/validation accuracy and loss curves

---

## 📊 Dataset

The model is trained on a chest X-ray image dataset organized into `train` and `test` directories, each containing three class subfolders:

| Class | Description |
|---|---|
| `COVID-19` | X-rays from patients diagnosed with COVID-19 |
| `Normal` | X-rays from healthy patients |
| `Viral Pneumonia` | X-rays from patients with non-COVID viral pneumonia |

**Dataset split:**
- Training set: 4,631 images (90%)
- Validation set: 513 images (10%, split from training data)
- Test set: 1,288 images

Images are resized to **224×224** pixels with 3 color channels before being fed into the network.

---

## 🏗️ Model Architecture

**PDCOVIDNet** ("Parallel Dilated COVID-Net") uses a two-branch design inspired by multi-scale feature extraction:

### Standard Convolution Branch (dilation rate = 1)
A stack of 5 convolutional blocks with increasing filter depth (64 → 128 → 256 → 512 → 512), each block containing:
- Two `Conv2D` layers (3×3 kernels) with ReLU activation
- A `MaxPooling2D` layer to downsample spatial dimensions

### Expanded Field Branch (dilation rate = 2)
A parallel stack of 5 identical-depth blocks, but using **dilated convolutions** (dilation rate = 2) to capture a wider receptive field without increasing parameter count as much as larger kernels would.

### Feature Fusion & Classification
- The final feature maps from both branches (7×7×512 each) are **concatenated** along the channel axis
- A fusion `Conv2D` layer (512 filters) processes the combined features
- The result is flattened and passed through two dense layers (1,024 units each) with ReLU activation and dropout (0.3) for regularization
- A final **softmax** output layer produces probabilities across the 3 classes

**Model size:** ~50.3 million trainable parameters (~191.8 MB)

The intuition behind this design is that the standard branch captures fine-grained local texture (important for detecting subtle opacities), while the dilated branch captures broader spatial context (useful for understanding overall lung field patterns) — combining both should improve diagnostic accuracy over a single-branch CNN.

---

## 🔄 Data Augmentation

To reduce overfitting and improve generalization on a relatively small medical imaging dataset, the training pipeline applies the following augmentations via Keras' `ImageDataGenerator`:

- **Rescaling:** Pixel values normalized to [0, 1]
- **Rotation:** Random rotations up to ±30°
- **Width/Height shift:** Random translation up to 15% of image dimensions
- **Shear:** Random shear transformations (10%)
- **Zoom:** Random zoom (10%)
- **Horizontal flip:** Random left-right flipping
- **Fill mode:** `nearest` interpolation for pixels introduced by transformations

Test data is only rescaled (no augmentation) to ensure evaluation reflects real-world image conditions.

---

## ⚙️ Training Setup

| Hyperparameter | Value |
|---|---|
| Image dimensions | 224 × 224 × 3 |
| Batch size | 32 |
| Number of classes | 3 |
| Learning rate | 1e-4 (initial) |
| Max epochs | 50 |
| Optimizer | Adam |
| Loss function | Categorical cross-entropy |

**Callbacks used:**
- **ModelCheckpoint** — saves the best model (by validation loss) to `pdcovidnet_best_model.h5`
- **ReduceLROnPlateau** — reduces the learning rate by a factor of 0.2 (down to a minimum of 1e-6) when validation loss plateaus for 3 epochs, helping the model converge more precisely in later training stages

---

## 📈 Results

After 50 training epochs with learning-rate decay:

| Metric | Validation | Test |
|---|---|---|
| **Accuracy** | 93.96% | 94.18% |
| **Loss** | 0.1996 | 0.1513 |

Training and validation accuracy/loss curves (plotted at the end of the notebook) show the model converging steadily, with the learning-rate reduction callback helping stabilize validation loss in later epochs.

---

## 🧰 Tech Stack

- **Language:** Python 3
- **Deep Learning Framework:** TensorFlow / Keras
- **Data Handling:** NumPy
- **Visualization:** Matplotlib
- **Environment:** Google Colab (trained on an A100 GPU with high-RAM runtime)

---

## 📁 Project Structure

```
pdcovidnet/
│
├── PDCOVIDNet.ipynb           # Main notebook: data pipeline, model, training, evaluation
├── pdcovidnet_best_model.h5   # Saved best model checkpoint (by validation loss)
├── data/
│   ├── train/
│   │   ├── COVID-19/
│   │   ├── Normal/
│   │   └── Viral Pneumonia/
│   └── test/
│       ├── COVID-19/
│       ├── Normal/
│       └── Viral Pneumonia/
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install tensorflow numpy matplotlib
```

### Running the Project

1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/pdcovidnet.git
   cd pdcovidnet
   ```
2. Organize your chest X-ray dataset into `data/train/<class>/` and `data/test/<class>/` folders as described above.
3. Update the dataset path variables in the notebook (`dataset_root`, `training_folder`, `testing_folder`) to point to your local data directory.
4. Open and run the notebook:
   ```bash
   jupyter notebook PDCOVIDNet.ipynb
   ```

> **Note:** Training this model is computationally intensive (~2 minutes per epoch on an A100 GPU). A GPU runtime is strongly recommended; CPU-only training will be substantially slower.

---

## 💡 Key Insights

- Combining **standard and dilated convolution branches** allows the network to capture both fine local detail and broader spatial context from a single input image, which is valuable in medical imaging where pathology can appear at multiple scales.
- **Data augmentation** was essential given the relatively small dataset size, helping the model generalize rather than memorize.
- The **learning-rate reduction schedule** played a key role in the final accuracy gains — most improvement in later epochs came after each LR reduction.
- The model reached strong performance (~94% test accuracy) on a 3-class classification task, though further validation on external, held-out datasets would be needed before considering any real-world clinical application.

---

## 🔮 Future Improvements

- Add **k-fold cross-validation** to get a more robust estimate of generalization performance.
- Generate a full **classification report and confusion matrix** on the test set (per-class precision/recall/F1), since accuracy alone doesn't reveal class-specific weaknesses.
- Apply **Grad-CAM** or similar visualization techniques to interpret which regions of the X-ray the model focuses on — important for clinical trust and validation.
- Experiment with **transfer learning** (e.g., fine-tuning a pretrained ImageNet backbone) as a baseline comparison against this custom architecture.
- Convert the saved model to the modern **Keras native format** (`.keras`) instead of the legacy HDF5 format for long-term compatibility.
- Test robustness against **out-of-distribution images** and different X-ray machine sources to check for dataset bias.

---

## ⚠️ Disclaimer

This project is intended for **educational and research purposes only**. It is not a validated medical diagnostic tool and should not be used for actual clinical decision-making without rigorous validation, regulatory approval, and oversight from qualified medical professionals.

---

## 📄 License

This project is open-source and available for educational and research purposes. Please cite appropriately if reused.
