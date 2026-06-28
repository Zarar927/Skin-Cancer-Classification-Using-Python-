# 🩺 Skin Cancer Classification using Deep Learning (CNN)

---

## 📌 Project Overview

Skin cancer is one of the most common forms of cancer worldwide. Early detection significantly increases the chances of successful treatment.

This project presents a **Convolutional Neural Network (CNN)** model built using **TensorFlow and Keras** to classify skin lesion images into two categories:

* 🟢 Benign (Non-Cancerous)
* 🔴 Malignant (Cancerous)

The model is trained on dermoscopic skin images and uses deep learning techniques such as convolutional layers, pooling layers, dropout, and checkpointing to achieve reliable performance.

---

## 🎯 Objectives

* Build an automated skin cancer detection system
* Perform image preprocessing and normalization
* Design and train a CNN model
* Save the best model during training
* Evaluate performance using accuracy and loss graphs

---

## 🛠️ Technologies Used

* Python 3.12
* TensorFlow / Keras
* NumPy
* Matplotlib
* ImageDataGenerator

---

## 📂 Dataset Structure

The dataset must be organized as:

```
skin_dataset_resized/
│
├── train_set/
│   ├── Benign/
│   └── Malignant/
│
└── val_set/
    ├── Benign/
    └── Malignant/
```

Update dataset paths in code:

```python
train_dir = r'path_to_train_set'
validation_dir = r'path_to_validation_set'
```

---

## 📷 Image Preprocessing

Images are processed using `ImageDataGenerator`.

### Steps:

* Rescaling pixel values (0–255 → 0–1)
* Resizing images to **150 × 150**
* Batch size = 32
* Binary classification mode

```python
ImageDataGenerator(rescale=1./255)
```

---

## 🧠 CNN Architecture

The model is built layer by layer:

| Layer      | Description                     |
| ---------- | ------------------------------- |
| Input      | 150 × 150 × 3 RGB images        |
| Conv2D     | 32 filters, 3×3, ReLU           |
| MaxPooling | 2×2                             |
| Conv2D     | 64 filters, 3×3, ReLU           |
| MaxPooling | 2×2                             |
| Conv2D     | 128 filters, 3×3, ReLU          |
| MaxPooling | 2×2                             |
| Conv2D     | 256 filters, 3×3, ReLU          |
| MaxPooling | 2×2                             |
| Conv2D     | 512 filters, 3×3, ReLU          |
| MaxPooling | 2×2                             |
| Flatten    | Converts feature maps to vector |
| Dense      | 128 neurons, ReLU               |
| Dropout    | 50%                             |
| Output     | 1 neuron, Sigmoid               |

---

## ⚙️ Model Compilation

```python
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)
```

### Settings:

* Optimizer: Adam
* Learning Rate: 0.0001
* Loss Function: Binary Crossentropy
* Metric: Accuracy

---

## 💾 Model Checkpoint

Best model is saved automatically:

```python
ModelCheckpoint(
    'best_model.keras',
    save_best_only=True,
    monitor='val_loss',
    mode='min'
)
```

---

## 🚀 Model Training

```python
history = model.fit(
    train_generator,
    epochs=15,
    validation_data=validation_generator,
    callbacks=[checkpoint]
)
```

### Training Configuration:

* Epochs: 15
* Batch Size: 32
* Validation Data: Yes
* Best Model Saved Automatically

---

## 📊 Training Results

* Training Accuracy: **~89%**
* Validation Accuracy: **~88%**
* Lowest Validation Loss: **0.2033**
* Best Model: `best_model.keras`

The model successfully learned patterns in dermoscopic images for distinguishing between benign and malignant skin lesions.

---

## 📈 Performance Visualization

### Accuracy Graph

```python
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
```

### Loss Graph

```python
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
```

These graphs help monitor:

* Learning progress
* Overfitting/underfitting
* Model stability

---

## 📌 Output Prediction

The model predicts:

* 🟢 Benign
* 🔴 Malignant

Final trained model file:

```
best_model.keras
```

This can be loaded later for real-time predictions.

---

## 📋 Future Improvements

* Apply data augmentation (rotation, flipping, zooming)
* Use transfer learning (EfficientNet, ResNet50, MobileNetV2)
* Add early stopping
* Increase dataset size
* Tune hyperparameters
* Add evaluation metrics:

  * Confusion Matrix
  * Precision, Recall, F1-score
  * ROC-AUC
* Deploy using Flask or Streamlit web app

---

## 📌 Conclusion

This project demonstrates a CNN-based approach for skin cancer classification. The model achieves strong performance with approximately:

* **89% Training Accuracy**
* **88% Validation Accuracy**

With further improvements like transfer learning and data augmentation, the system can become even more accurate and suitable for real-world medical applications.

---

## ⭐ If you like this project

Give it a ⭐ on GitHub and feel free to contribute or improve it!
