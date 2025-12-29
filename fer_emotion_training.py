# FER-2013 Emotion Recognition Training Script
# This script trains a CNN model on the FER-2013 dataset using TensorFlow/Keras.
# Designed for Google Colab environment.

# %%
# Import necessary libraries
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

# %%
# Define emotion labels
emotion_labels = {0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy", 4: "Sad", 5: "Surprise", 6: "Neutral"}

# %%
# Load FER-2013 dataset from image folders
# Assuming the dataset is extracted to C:/Users/shreyash/Downloads/archive/train and test
train_dir = 'C:/Users/shreyash/Downloads/archive/train'
test_dir = 'C:/Users/shreyash/Downloads/archive/test'

if not os.path.exists(train_dir):
    print(f"{train_dir} not found. Please ensure the FER-2013 dataset is extracted to the correct path.")
    raise FileNotFoundError(f"{train_dir} not found.")

if not os.path.exists(test_dir):
    print(f"{test_dir} not found. Please ensure the FER-2013 dataset is extracted to the correct path.")
    raise FileNotFoundError(f"{test_dir} not found.")

# %%
# Data generators for train and test
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2  # Split train into train and validation
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(48, 48),
    batch_size=64,
    color_mode='grayscale',
    class_mode='categorical',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(48, 48),
    batch_size=64,
    color_mode='grayscale',
    class_mode='categorical',
    subset='validation'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(48, 48),
    batch_size=64,
    color_mode='grayscale',
    class_mode='categorical',
    shuffle=False
)

# Data augmentation is already included in train_datagen

# %%
# Build CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(7, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# %%
# Callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
checkpoint = ModelCheckpoint('model.h5', monitor='val_accuracy', save_best_only=True)

# %%
# Train the model
history = model.fit(
    train_generator,
    epochs=50,
    validation_data=validation_generator,
    callbacks=[early_stop, checkpoint]
)

# %%
# Plot training accuracy & loss curves
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.show()

# %%
# Evaluate on test set
test_loss, test_accuracy = model.evaluate(test_generator)
print(f"Test Accuracy: {test_accuracy}")

# Predictions
y_pred = model.predict(test_generator)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = test_generator.classes

print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred_classes))
print("\nClassification Report:")
print(classification_report(y_true, y_pred_classes, target_names=list(emotion_labels.values())))

# %%
# Save trained model as model.h5 (already saved via checkpoint)
# Save as TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open('emotion_model.tflite', 'wb') as f:
    f.write(tflite_model)

# %%
# Inference function
def predict_emotion(image_path):
    # Load and preprocess image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Image not found or unable to load")
    img = cv2.resize(img, (48, 48))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=[0, -1])  # Shape: (1, 48, 48, 1)

    # Predict
    prediction = model.predict(img)
    emotion_index = np.argmax(prediction)
    return emotion_labels[emotion_index]

# %%
# Test inference using a sample image
# Assuming a sample image is uploaded, e.g., 'sample_face.jpg'
# Replace with actual path
sample_image_path = 'sample_face.jpg'  # Upload this in Colab
if os.path.exists(sample_image_path):
    predicted_emotion = predict_emotion(sample_image_path)
    print(f"Predicted Emotion for sample image: {predicted_emotion}")
else:
    print("Sample image not found. Please upload 'sample_face.jpg' to test inference.")
