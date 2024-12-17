# 🧠 Optimized U-Net for Brain Tumor Segmentation

---

## 🚀 **Project Overview**

This project implements an **Optimized U-Net Architecture** to perform precise **brain tumor segmentation** on MRI images. Designed to process the **BraTS 2020 Dataset**, the model achieves exceptional results in identifying tumor subregions like **edema**, **non-enhancing tumors**, and **enhancing tumors**.

---

## 📊 **Dataset Details**

| **Property**            | **Details**                     |
|--------------------------|---------------------------------|
| **Dataset Name**         | BraTS 2020                     |
| **Image Modalities**     | T1, T1ce, T2, FLAIR (MRI)      |
| **Data Format**          | NIfTI (.nii files)             |
| **Image Resolution**     | 240 x 240 x 155                |
| **Classes**              | Background, Edema, Non-enhancing Tumor, Enhancing Tumor |

---

## ✨ **Project Features**

- 📈 **High Accuracy:** ~99% segmentation accuracy  
- 🛠️ **Advanced U-Net:** Optimized with additional layers and activations  
- 🔄 **Efficient Training:** Runs for 40 epochs using Adam Optimizer  
- 🎯 **Metrics Tracked:** Dice Coefficient, Accuracy, Sensitivity, and Specificity  
- 🧩 **Modular Design:** Clear separation of data loading, model training, and evaluation  

---

## 🔧 **System Setup**

### 🖥️ **Hardware Requirements**  
- **RAM:** Minimum 4GB  
- **Disk Space:** 200GB (for MRI dataset storage)  
- **GPU:** Recommended (NVIDIA P100 or higher)  

### 🛠️ **Software Requirements**  
| **Software/Library** | **Version**    |
|-----------------------|----------------|
| Python               | 3.7+           |
| TensorFlow/Keras     | 2.15.0         |
| scikit-learn         | 1.4.1.post1    |
| NumPy                | 1.26.4         |
| Matplotlib           | 3.8.3          |
| OpenCV (cv2)         | 4.8.0          |
| NiBabel              | 5.2.1          |
| scikit-image         | 0.22.0         |

---

## 🔧 **Methodology**

### 1. **Data Preprocessing**
   - Convert NIfTI MRI scans into a usable format (e.g., 2D slices).
   - Perform **skull-stripping** to remove unnecessary parts of the MRI images.
   - Apply **intensity normalization** to standardize pixel intensity across images.
   - Resize all MRI slices to `240x240` resolution for uniform input.

### 2. **Model Design**
   - Implement a **U-Net Architecture**:
      - **Encoder Path (Contraction):** Downsampling using convolutional layers followed by MaxPooling.
      - **Bottleneck Layer:** Captures the image's most abstract features.
      - **Decoder Path (Expansion):** Upsampling to restore image dimensions and concatenate with encoder features for better localization.
   - Use **ReLU** activation for convolutional layers.
   - Add **Batch Normalization** to speed up convergence and stabilize training.

### 3. **Training**
   - **Optimizer:** Adam optimizer with a learning rate of `0.001`.
   - **Loss Function:** Binary cross-entropy loss for pixel-wise classification.
   - **Metrics:** Dice Coefficient, Accuracy, Precision, Recall, Sensitivity, and Specificity.
   - **Epochs:** 40
   - **Batch Size:** 32
   - Apply **EarlyStopping** to prevent overfitting.

### 4. **Evaluation**
   - Evaluate the model on unseen MRI slices.
   - **Metrics Used:**
      - **Accuracy**
      - **Dice Coefficient**
      - **Specificity**
      - **Sensitivity**
   - Visualize predictions against ground truth masks.

### 5. **Prediction**
   - Run the trained model on new MRI images.
   - Generate **segmented masks** for brain tumor regions:
      - **Edema**
      - **Non-enhancing Tumor**
      - **Enhancing Tumor**

---

### 🚀 **Pipeline Overview**
1. Data Extraction → Preprocessing → Model Training → Evaluation → Prediction
2. Tools: Python, TensorFlow/Keras, OpenCV, Nibabel, Scikit-learn
3. Results: High segmentation accuracy (~99%) on the BraTS 2020 dataset.

---

## 🚀 **Results**

- **Accuracy:** `99.07%`
- **Dice Coefficient:** `0.9707`
- **Specificity:** `99.87%`
- **Sensitivity:** `97.07%`

### **Model Comparison Table**

| **Model**            | **Accuracy** | **Sensitivity** | **Specificity** |
|-----------------------|--------------|-----------------|-----------------|
| R-CNN (2020)         | 94.1%        | 72%             | -               |
| AlexNET (2019)       | 96.1%        | 95.2%           | 95.1%           |
| **Proposed Model**    | **99.07%**   | **97.07%**      | **99.87%**      |

### **Segmentation Output Example**

| **MRI Input**        | **Ground Truth**      | **Predicted Output**    |
|-----------------------|-----------------------|-------------------------|
| ![Input](image1.png)  | ![Truth](mask1.png)   | ![Output](output1.png)  |

---
