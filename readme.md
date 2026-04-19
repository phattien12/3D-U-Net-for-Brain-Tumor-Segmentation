# 🧠 3D U-Net: Multimodal Brain Tumor Segmentation

A high-performance deep learning pipeline for **3D volumetric brain tumor segmentation** using a lightweight **3D U-Net** architecture. This project leverages the **BraTS 2020** dataset to perform multi-class voxel-wise classification across four MRI modalities.

[](https://www.python.org/)
[](https://tensorflow.org/)
[](https://www.kaggle.com/awsaf49/brats20-dataset-training-validation)
[](https://opensource.org/licenses/MIT)

-----

## 📌 Overview

Manual annotation of brain tumors by radiologists is a labor-intensive process prone to inter-observer variability. This project automates **multi-class segmentation**, identifying the tumor core, peritumoral edema, and enhancing tumor regions using a **3D Convolutional Neural Network (CNN)**.

### ✨ Key Features

  * **Full 3D Pipeline**: Processes volumetric data directly, preserving spatial context across the Z-axis.
  * **Multimodal Input**: Integrates FLAIR, T1, T1CE, and T2 MRI sequences.
  * **Hybrid Loss Function**: Combines **Categorical Cross-Entropy** with **Dice Loss** to handle class imbalance.
  * **Advanced Visualization**: Includes 2D slice comparisons and **3D interactive rendering** of predicted tumors.

-----

## 📂 Dataset: BraTS 2020

The **Multimodal Brain Tumor Segmentation Challenge (BraTS)** dataset provides gold-standard labels for brain tumor sub-regions.

### MRI Modalities Used

1.  **FLAIR**: Fluid Attenuated Inversion Recovery (highlights Edema).
2.  **T1 / T1CE**: T1-weighted and Contrast-Enhanced (highlights Tumor Core).
3.  **T2**: T2-weighted (highlights general fluid/anatomy).

### Class Mapping

The labels are converted for training convenience:
| Original Label | Class ID | Meaning |
| :--- | :--- | :--- |
| 0 | 0 | Background |
| 1 | 1 | Necrotic / Non-enhancing Tumor Core |
| 2 | 2 | Peritumoral Edema |
| 4 | 3 | Enhancing Tumor |

-----

## 🏗️ Model Architecture: 3D U-Net

The architecture is a lightweight 3D variant of the classic U-Net, optimized for medical volumetric data.

  * **Encoder Path**: Successive levels with 8, 16, and 32 filters. Uses `Conv3D`, `BatchNorm`, and `ReLU`.
  * **Bottleneck**: 64 filters to capture high-level latent features.
  * **Decoder Path**: Transposed convolutions for upsampling, concatenated with high-resolution features from the encoder via **Skip Connections**.
  * **Output**: Softmax activation providing voxel-wise probability maps for 4 classes.

-----

## ⚙️ Data Preprocessing

1.  **Normalization**: Every modality is scaled to $[0, 1]$ to ensure numerical stability.
2.  **Resizing**: Input volumes ($240 \times 240 \times 155$) are resized to **$128 \times 128 \times 128$**.
      * *MRI Sequences*: Bilinear interpolation.
      * *Masks*: Nearest Neighbor interpolation (to preserve label integrity).
3.  **Channel Stacking**: The 4 modalities are stacked into a 4-channel 3D tensor: $(128, 128, 128, 4)$.

-----

## 🏋️ Training Configuration

| Parameter | Value |
| :--- | :--- |
| **Optimizer** | Adam ($\text{LR} = 10^{-4}$) |
| **Loss** | $\text{Categorical Cross Entropy} + \text{Dice Loss}$ |
| **Batch Size** | 1 (due to high VRAM consumption of 3D tensors) |
| **Epochs** | 30 |
| **Precision** | Mixed Float16 (for faster training on Tesla T4/A100) |

-----

## 📊 Performance Results

The model achieves high voxel-wise accuracy, though the Dice score reflects the high difficulty of segmenting small tumor boundaries with limited training samples.

  * **Train Accuracy**: 97.74%
  * **Validation Accuracy**: 97.48%
  * **Dice Coefficient**: \~0.48 (Balanced across all classes)

-----

## 🎨 Visualization

### 1\. 2D Multi-modal Slice

Comparison of FLAIR, T1, T1CE, and T2 sequences alongside the ground truth segmentation.

### 2\. 3D Tumor Reconstruction

Using `Plotly` and `Scikit-image`, the project renders the 3D structure of the tumor:

  * 🔴 **Red**: Necrosis
  * 🔵 **Blue**: Edema
  * 🟢 **Green**: Enhancing Tumor

-----

## 🚀 Installation & Usage

### Prerequisites

```bash
pip install nibabel opencv-python scikit-learn matplotlib plotly scikit-image tensorflow
```

### Running the Project

1.  **Clone the Repo**:
    ```bash
    git clone https://github.com/yourusername/brats-3d-unet.git
    ```
2.  **Setup Data**: Place your BraTS 2020 `.nii` files in the specified directory or connect to Kaggle.
3.  **Execute**: Run the Jupyter Notebook `3D U-Net for Brain Tumor Segmentation.ipynb`.

-----

## 🛠️ Roadmap & Limitations

**Current Limitations:**

  * Trained on a subset (20 patients) for demonstration purposes.
  * Lightweight filter count (8-64) to fit on commercial GPUs.

**Future Enhancements:**

  - [ ] Integration with **MONAI** framework.
  - [ ] **Attention U-Net** or **Residual U-Net** layers.
  - [ ] Patch-based training to handle original $240 \times 240 \times 155$ resolution.
  - [ ] Data augmentation (rotation, flipping, elastic deformation).

-----

## 👨‍💻 Author

**Phat**
*Research Interests: Deep Learning in Healthcare, Medical Imaging, and Computer Vision.*

-----

*Disclaimer: This tool is for research and educational purposes only. It is not intended for clinical diagnostic use.*