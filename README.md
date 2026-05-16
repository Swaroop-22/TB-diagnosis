
# Tuberculosis (TB) Diagnosis using Deep Learning

This repository contains an end-to-end Computer Vision and Deep Learning framework designed to detect and diagnose Tuberculosis (TB) from medical imaging datasets (such as chest X-rays). The system leverages advanced Convolutional Neural Networks (CNNs) to automate binary classification, acting as a reliable decision-support system for clinical radiologists.

---

## 📌 Project Architecture & Pipeline

The framework implements a robust deep learning pipeline to ensure high sensitivity and specificity:

1. **Data Augmentation & Preprocessing:** Standardizes raw medical images via resizing, normalization, and geometric augmentations (rotation, zoom, horizontal flips) to minimize overfitting.
2. **Transfer Learning Architecture:** Utilizes deep convolutional features by adapting state-of-the-art models (such as ResNet, VGG, or MobileNet) pre-trained on ImageNet.
3. **Model Fine-Tuning:** Customizes the final classification head to output the probability of TB infection (Normal vs. Tuberculosis).
4. **Diagnostic Evaluation:** Measures performance using critical medical diagnostic metrics, including Confusion Matrices, ROC-AUC curves, Sensitivity, and Specificity.

---

## 🛠️ Installation & Dependencies

To execute the training notebook or deploy the model locally, configure your environment with the following dependencies:

```bash
pip install numpy pandas matplotlib seaborn opencv-python tensorflow keras scikit-learn

```

> **Note:** For accelerated model training and evaluation, running this pipeline in an environment with GPU capabilities (CUDA enabled) is highly recommended.

---

## 📊 Dataset Structure

The pipeline expects a structured image directory partitioned for training, validation, and testing:

```text
dataset/
├── train/
│   ├── Normal/
│   └── Tuberculosis/
├── val/
│   ├── Normal/
│   └── Tuberculosis/
└── test/
    ├── Normal/
    └── Tuberculosis/

```

---

## 💻 Code Implementation & Model Development

### 1. Data Pipeline Configuration

TensorFlow/Keras data generators are utilized to stream and augment images efficiently from disk:

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Initialize augmentation parameters
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_generator = train_datagen.flow_from_directory(
    'dataset/train',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

```

### 2. Model Pipeline Architecture

The script builds a custom classifier on top of a frozen convolutional backbone:

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import ResNet50

# Load pre-trained base model
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Freeze initial weights

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Binary decision output
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

```

---

## 📈 Diagnostic Metrics & Results

Model performance is evaluated across several clinical standards:

* **Accuracy Score:** Overall correct classification rate.
* **Sensitivity (Recall):** The ability of the model to correctly identify true positive TB cases (critical in medical diagnostics to prevent false negatives).
* **Confusion Matrix:** Tracks True Positives (TP), False Positives (FP), True Negatives (TN), and False Negatives (FN).
* **ROC-AUC Score:** Evaluates performance across all possible classification thresholds.

```python
from sklearn.metrics import classification_report, confusion_matrix

# Evaluating model on unseen test data
y_pred = (model.predict(test_generator) > 0.5).astype("int32")
print(classification_report(test_generator.classes, y_pred, target_names=['Normal', 'Tuberculosis']))

```

---

## 🔮 Future Enhancements

* **Grad-CAM Integration:** Introduce Gradient-weighted Class Activation Mapping to visually highlight the specific regions of the chest X-ray driving the model's positive TB predictions.
* **Multi-Class Expansion:** Train the model to differentiate TB from other structurally similar pulmonary infections like pneumonia or bronchitis.
* **Web App Deployment:** Build a lightweight Streamlit or Flask interface to allow real-time image uploads and immediate inference generation.

```

```
