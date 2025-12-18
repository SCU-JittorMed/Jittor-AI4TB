
# jittorMedTB: Efficient Medical Image Segmentation with Jittor

**jittorMedTB** is a lightweight, high-performance medical image segmentation framework built on **[Jittor](https://github.com/Jittor/jittor)**.

Inspired by the design philosophy of nnU-Net, this project implements efficient training and inference pipelines for both **3D Volumetric (CT)** and **2D Slice-based (DR/X-Ray)** segmentation tasks, specifically optimized for Liver and Tumor segmentation.

## 📂 Project Structure

The project is organized into two main modules based on data dimensionality:

```text
jittorMedTB/
├── assets/                 # Images and static assets
├── jittorMedTB_CT/         # 3D Volumetric Segmentation Module
│   ├── train.py            # 3D U-Net Training (with resume & logging)
│   └── predict.py          # 3D Sliding Window Inference (NIfTI support)
├── jittorMedTB_DR/         # 2D Slice-based Segmentation Module
│   ├── train2d.py          # 2D U-Net Training (On-the-fly 3D-to-2D slicing)
│   └── predict2d.py        # 2D Inference script
└── README.md
```

## ✨ Key Features

  * **🚀 Jittor JIT Acceleration**: Leverages Jittor's Just-In-Time compilation for high-performance operator fusion without writing CUDA code.
  * **🛠️ Robust Environment Support**: Solves compatibility issues between high-version Linux (Ubuntu 22.04) and legacy CUDA (11.x) using specific GCC configurations.
  * **🔄 Smart Training**:
      * **Auto-Resume**: Automatically detects existing checkpoints and resumes training after interruptions.
      * **Logging**: Dual logging (Console + TXT) aligned with nnU-Net style metrics.
  * **🧠 Advanced Inference**: Implements **Sliding Window Inference** for large 3D volumes to optimize memory usage.
  * **🔪 On-the-fly Slicing**: The 2D module supports real-time slicing from 3D data with **foreground oversampling** strategies.

## ⚙️ Prerequisites & Installation

**This is the most critical step.** To ensure Jittor compiles operators correctly on modern Linux systems, please follow these configuration steps strictly.

### 1\. System Requirements

  * **OS**: Ubuntu 20.04 / 22.04
  * **Python**: 3.8+
  * **CUDA**: 11.5+ (Recommended)
  * **GCC**: **Version 10** (Crucial for CUDA 11 compatibility on newer Ubuntu)

### 2\. Install Dependencies

```bash
# 1. Install Python libraries
pip install jittor numpy nibabel

# 2. Install GCC-10 (Required for Jittor JIT)
sudo apt update
sudo apt install -y g++-10
```

### 3\. Configure Environment Variables

Before running any scripts, **you must export the compiler path**:

```bash
export cc_path=/usr/bin/g++-10
export nvcc_path=/usr/local/cuda/bin/nvcc 
# Note: Jittor usually finds nvcc automatically, but setting cc_path is mandatory.
```

## 🚀 Usage

### Data Preparation

This project uses the `.npy` format preprocessed by nnU-Net. Please ensure your data is located in the path defined in `DATA_DIR` inside the scripts (default: `/data/syg/...`).

### 1\. 3D Segmentation (CT Module)

Enter the CT directory to train the 3D U-Net model.

```bash
cd jittorMedTB_CT

# Start Training (3D)
# Logs will be saved to 'training_log.txt'
python train.py

# Inference (Prediction)
# Modify MODEL_PATH and INPUT_PATH in predict.py before running
python predict.py
```

### 2\. 2D Segmentation (DR Module)

Enter the DR directory to train the 2D U-Net model. This module reads 3D volumes and extracts random Z-slices during training.

```bash
cd jittorMedTB_DR

# Start Training (2D)
python train2d.py

# Inference (2D)
python predict2d.py
```

## 📊 Logging & Results

The training logs provide detailed metrics including Soft Dice Loss and Pseudo Dice, saved in real-time.

**Example Log Output:**

```text
2025-12-16 17:53:26.378: Epoch 2
2025-12-16 17:53:26.378: Current learning rate: 0.0010
2025-12-16 17:53:26.378: train_loss -0.0173
2025-12-16 17:53:26.378: val_loss -0.0228
2025-12-16 17:53:26.378: Pseudo dice [0.0338]
```

## 🛠️ Troubleshooting

**Q: `fatal error: stdlib.h: ... __attribute_alloc_align__`**

> **A:** This is due to a GCC version mismatch. Ensure you installed `g++-10` and ran `export cc_path=/usr/bin/g++-10`.

**Q: `file cudnn.h not found`**

> **A:** Jittor requires the cuDNN development library. You can let Jittor install it automatically:
>
> ```bash
> python -m jittor_utils.install_cuda
> ```

**Q: `KeyError: 'model_state'` during inference**

> **A:** The `predict.py` script includes an auto-detection mechanism. It supports loading both raw state dictionaries and full checkpoints (containing optimizer states).

## 📜 License

This project is released under the MIT License.