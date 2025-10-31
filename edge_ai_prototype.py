"""
edge_ai_prototype.py

Author: Amanuel Alemu Zewdu
Purpose: Demonstrate Edge AI workflow using TensorFlow Lite simulation on image classification.
Requirements: tensorflow, numpy, matplotlib
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# Simulate dataset
X_train = np.random.rand(100, 32, 32, 3)
y_train = np.random.randint(0, 2, 100)
X_test = np.random.rand(20, 32, 32, 3)
y_test = np.random.randint(0, 2, 20)

# Simple CNN model
model = models.Sequential([
    layers.Conv2D(16, (3,3), activation='relu', input_shape=(32,32,3)),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(16, activation='relu'),
    layers.Dense(2, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=2, verbose=1)

# Evaluate
loss, acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {acc:.3f}")

# Convert to TensorFlow Lite model
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open("edge_model.tflite", "wb") as f:
    f.write(tflite_model)

print("Edge AI model converted to TensorFlow Lite.")
