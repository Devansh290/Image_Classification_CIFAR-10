# 🧠 Image Classification with CNN on CIFAR-10 Dataset

This project implements a **Convolutional Neural Network (CNN)** from scratch using **TensorFlow and Keras** to classify images from the **CIFAR-10 dataset**, which consists of 60,000 32x32 color images in 10 categories. The model achieves ~77% accuracy and uses data augmentation to improve generalization.

---

## 📌 Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training and Evaluation](#training-and-evaluation)
- [Results](#results)
- [Limitations](#limitations)
- [Future Scope](#future-scope)
- [Use Cases](#use-cases)
- [Conclusion](#conclusion)

---

## 📊 Overview

The goal of this project is to classify images into one of 10 categories using a CNN model trained on the CIFAR-10 dataset. The model is built using Keras Sequential API with convolutional, pooling, dense layers, and uses softmax for multi-class classification. Data augmentation techniques are applied to improve performance.

---

## 🗂️ Dataset

- **CIFAR-10**: 60,000 32x32 RGB images in 10 classes (6,000 images per class)
- **Train/Test Split**: 50,000 training and 10,000 test images

---

## 🧠 Model Architecture

The CNN is implemented using the following architecture:

- `Conv2D` (32 filters, 3x3, ReLU)
- `MaxPooling2D` (2x2)
- `Conv2D` (64 filters, 3x3, ReLU)
- `MaxPooling2D` (2x2)
- `Flatten`
- `Dense` (64 units, ReLU)
- `Dense` (10 units, softmax)

**Optimizer**: Adam  
**Loss**: Categorical Crossentropy  
**Metric**: Accuracy

---

## 🔁 Training and Evaluation

### ✅ Preprocessing
- Normalized pixel values (0–1)
- One-hot encoded class labels

### 🔁 Data Augmentation
Using `ImageDataGenerator`:
- Rotation range: 15°
- Width/height shift: 10%
- Horizontal flip: Enabled

### 🏋️ Training
- Epochs: 50
- Batch size: 64
- Trained on augmented data
- Validated on test set

---

## 📈 Results

- **Test Accuracy**: ~77%
- **Training/Validation Plots**:
  - Accuracy steadily increases
  - Loss decreases with more epochs
- **Model saved as**: `cifar10_model.h5`

---

## ⚠️ Limitations

- **Low resolution** of CIFAR-10 images
- **Limited model complexity** may cause underfitting
- **No hyperparameter tuning**
- **Data augmentation limited** to basic transformations

---

## 🔭 Future Scope

- Add **more layers or deeper CNN**
- Use **transfer learning** with models like ResNet or VGG
- Apply **advanced augmentation** (cutout, mixup)
- Perform **hyperparameter tuning**
- Use **ensemble learning** to improve results

---

## 🚀 Use Cases

1. **Autonomous Vehicles**: Object detection for traffic scenes
2. **E-commerce**: Product classification and recommendations
3. **Surveillance**: Suspicious object/person detection
4. **Healthcare**: Basic image-based anomaly classification
5. **Manufacturing**: Visual quality control in assembly lines

---

## ✅ Conclusion

This project demonstrates how a beginner-friendly CNN can be built from scratch using TensorFlow and Keras. Despite its simplicity, the model reached 77% accuracy. With further fine-tuning and enhancements, the approach can be extended to real-world use cases.

---
