# 🧠 Optimized U-NET Architecture For Brain Tumor Segmentation

### 🖥️ **Overview**

This project implements an optimized **U-Net architecture** for **Brain Tumor Segmentation** using MRI images. The U-Net model is trained on the **BraTS 2020** dataset and is capable of segmenting brain tumors with high accuracy, including different subregions such as edema, non-enhancing tumor, and enhancing tumor.

---

### 📊 **Dataset**
- **Dataset Name:** BraTS 2020
- **Modality:** MRI (T1, T1ce, T2, FLAIR)
- **Format:** NIfTI files
- **Resolution:** 240 x 240 x 155
- **Classes:** Background, Edema, Non-enhancing Tumor, Enhancing Tumor

---

### ✨ **Features**
- **High Accuracy:** Achieves ~99% accuracy in tumor segmentation.
- **Efficient Training:** Model trained for 40 epochs with batch size of 32.
- **Advanced Techniques:** Data augmentation, custom loss functions for better segmentation.
- **Small Dataset Handling:** U-Net is optimized to work with relatively small medical datasets.
- **Modular Design:** Code is split into modules for data extraction, model building, training, and evaluation.

---

### 💻 **System Specifications**
#### **Hardware**
- **RAM:** Minimum 4GB
- **Disk Space:** Minimum 200GB
- **GPU:** Recommended for faster training (P100 or better)
  
#### **Software**
- **Operating System:** Windows/Linux/MacOS
- **Python:** 3.7 or higher
- **Required Libraries:** 
  - TensorFlow/Keras
  - Sklearn
  - Skimage
  - Matplotlib
  - Glob
  - Nibabel

---

### 🔧 **Methodology**
1. **Data Preprocessing:** 
   - MRI scans are preprocessed (skull-stripping, intensity normalization, etc.) for consistency.
   
2. **Model Design:** 
   - U-Net architecture with 4 layers in contraction and expansion paths.
   - Convolution layers with ReLU activation, followed by max-pooling layers.
   
3. **Training:** 
   - Trained for 40 epochs with a batch size of 32.
   - Optimizer: Adam with a learning rate of 0.001.
   - Loss function: Binary cross-entropy.
   
4. **Evaluation:** 
   - Metrics used: Dice coefficient, accuracy, precision, recall, and specificity.
   - Achieved ~99% accuracy with a 1% loss.

---

### 📋 **Output**

<img src="https://github.com/user-attachments/assets/ed8e06eb-69b2-4185-9da0-b407ff0eb2f0" width="600">

### 🚀 **Results**

- **Accuracy:** `99.07%`
- **Dice Coefficient:** `0.9707`
- **Specificity:** `99.87%`
- **Sensitivity:** `97.07%`
  
Comparison with Other Models:

| Model               | Accuracy | Sensitivity | Specificity |
|---------------------|----------|-------------|-------------|
| R-CNN (2020)        | 0.941    | 0.72        | -           |
| AlexNET (2019)      | 0.961    | 0.952       | 0.951       |
| Proposed Model      | 0.9907   | 0.9707      | 0.9987      |

---

### 🛠️ **Installation**
1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/brain-tumor-segmentation.git
    ```
2. Navigate to the project directory:
    ```bash
    cd brain-tumor-segmentation
    ```
3. Create a virtual environment and activate it:
    ```bash
    python3 -m venv venv
    source venv/bin/activate 
    ```
4. Install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```

---

### 🧪 **Usage**
1. **Dataset Preparation:**
   - Download the BraTS 2020 dataset and place it in the `data/` directory.

2. **Import Dataset:**
    ```bash
    python Importing Data & Splitting.py
    ```
    
2. **Model Creation and Training:**
    ```bash
    python Model Creation.py
    python Model Training.py
    ```

3. **Model Evaluation:**
    ```bash
    python evaluate.py
    ```

4. **Segmentation on New Images:**
    ```bash
    python Prediction & Visualization.py # Set desired to path
    ```

---

### 👥 **Contributors**
- **Logeshwaran K S**
- **Email:** logeshwaranks01@gmail.com
  
---

### 📄 **License**
This project is licensed under the MIT License.
