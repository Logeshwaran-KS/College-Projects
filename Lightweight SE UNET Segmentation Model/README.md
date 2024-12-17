# 🧠 Brain Tumor Segmentation on MRI Images Using Lightweight SE UNET Architecture

---

## 🚀 **Project Overview**

This project implements an **Lightweight SE UNET Architecture** to perform precise **brain tumor segmentation** on MRI images. Designed to process the **T1 Weighted dataset**, the model achieves exceptional results in identifying tumor regions in less amount of time and memory.
---

## 📊 **Dataset Details**

| **Property**            | **Details**                     |
|--------------------------|---------------------------------|
| **Dataset Name**         | Figshare BT Dataset            |
| **Image Modalities**     | T1 Weighted                    |
| **Data Format**          | Image Format                   |
| **Image Resolution**     | 128 x 128 x 155                |

---

## ✨ **Project Features**

- 📈 **High Accuracy:** ~99.4% segmentation accuracy  
- 🛠️ **Lightweight SE U-Net:** Inclusion of SE block with Single Conv Layer 
- 🔄 **Efficient Training:** Runs for 50 epochs using Adam Optimizer  
- 🎯 **Metrics Tracked:** Dice Coefficient, Accuracy, Dice Loss, and Specificity  
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

- **Accuracy:** `99.39%`
- **Dice Coefficient:** `81.89%`
- **Specificity:** `99.7%`
- **Dice Loss:** `2%`

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

### 🛠️ **Installation**
1. Clone the repository:
    ```bash
    git clone https://github.com/Logeshwaran-KS/College-Projects/Lightweight-SE-UNET-Segmentation-Model.git
    ```
2. Navigate to the project directory:
    ```bash
    cd Lightweight-SE-UNET-Segmentation-Model
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
   - Download the Figshare T1 Weighted BT dataset and place it in the `data/` directory.

2. **Import Dataset:**
    ```bash
    python Importing Data & Splitting.py
    ```
    
2. **Model Creation and Training:**
    ```bash
    python Model Creation.py
    python Model Training.py
    ```

3. **Performance Visualization:**
    ```bash
    python Performance Visualization.py
    ```

4. **Segmentation on New Images:**
    ```bash
    python Prediction & Visualization.py # Set desired to path
    ```

---

### 👥 **Contributors**
- **Logeshwaran K S**
- **Kalluri Anisha Devi**
- **Email:** logeshwaranks01@gmail.com
- **Email:** a.anisha18102003@gmail.com
  
---

### 📄 **License**
This project is licensed under the MIT License.
