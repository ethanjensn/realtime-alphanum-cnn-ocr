import os
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from emnist import extract_training_samples, extract_test_samples
import emnist


# python CharCNN_train.py

# Check for GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("GPU is available. Using GPU:", gpus)
else:
    print("No GPU found. Using CPU.")

print("TensorFlow version:", tf.__version__)
print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))
print("GPU Devices:", tf.config.list_physical_devices('GPU'))

# # check to see if emnist data is cached
# print("Checking EMNIST data...")
# emnist.ensure_cached_data()


# manually download emnist dataset

cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "emnist")
cache_path = os.path.join(cache_dir, "emnist.zip")

# Updated link using 'biometrics'
url = "https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip"

if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)

if not os.path.exists(cache_path) or os.path.getsize(cache_path) < 500_000_000:
    print("Downloading real EMNIST dataset (~500MB)...")
    with requests.get(url, stream=True, verify=False) as r:
        with open(cache_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    print("Download complete!")

# load the training samples for the 'byclass' subset
train_images, train_labels = extract_training_samples('byclass')
test_images, test_labels = extract_test_samples('byclass')
print(f"Success! Loaded {len(train_images)} training images.")


def load_emnist_label_mapping():
    mapping = {}
    with open('../../training/emnist-byclass-mapping.txt', 'r') as f:
        for line in f:
            index, ascii_code = map(int, line.strip().split())
            mapping[index] = chr(ascii_code)
    return mapping

label_map = load_emnist_label_mapping()


# Normalize pixel values
train_images = train_images.reshape(-1, 28, 28, 1) / 255.0
test_images = test_images.reshape(-1, 28, 28, 1) / 255.0
# -1: Tells NumPy to infer the number of samples
print("Min pixel value:", train_images.min())
print("Max pixel value:", train_images.max())

# Note: Using CNN format, no need for flattened data
# train_images and test_images are already in (batch, 28, 28, 1) format

# CNN 2.0
model = keras.Sequential([
    # Input layer
    keras.layers.Input(shape=(28, 28, 1)),

    # Convolutional layers
    keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Dropout(0.25),

    keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Dropout(0.25),

    keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    keras.layers.BatchNormalization(),
    keras.layers.GlobalAveragePooling2D(),
    keras.layers.Dropout(0.5),

    # Output layer
    keras.layers.Dense(62)

    # Conv2D: "Find patterns in the image"
    # BatchNorm: "Clean up the data"
    # MaxPool: "Shrink the image, keep the important stuff"
    # Dropout: "Don't get too comfortable with any one pattern"

    # Using pooling, a lower resolution version of input is created that still contains
    # the large or important elements of the input image.
    # helping to preserve important information and features of the input image
])

model.compile(
    optimizer=tf.keras.optimizers.AdamW(
        learning_rate=0.001,
        weight_decay=0.01
    ),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy', 'sparse_top_k_categorical_accuracy']  # Use sparse version
)

# Learning rate scheduling
lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-7
)

# Early stopping
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

#RUN THIS TO TRAIN THE MODEL
# Then train with callbacks:
history = model.fit(
    train_images, train_labels,
    epochs=20,
    batch_size=128,
    validation_split=0.2,
    callbacks=[lr_scheduler, early_stop],
    verbose=1
)

# Load the trained model (choose the correct filename)
# model = keras.models.load_model('../../models/chars_and_digits_model.h5')

model.evaluate(test_images, test_labels)

# load all predicted values
y_predicted = model.predict(test_images)

y_predicted[4] # trying to predict the second image

label_map = load_emnist_label_mapping()  # loads the mapping of numbers to characters
# this is a dictionary that maps numbers to characters

y_predicted_labels = [np.argmax(i) for i in y_predicted]  # list of predicted indices
predicted_char_labels = [label_map[idx] for idx in y_predicted_labels]  # list of predicted characters
# for each index in the list of predicted indices, it finds the corresponding character in the label_map

# for each index in the list y_predicted_labels,
# Look up label_map[idx] (i.e., get the character for that index)
# then put that character in the list predicted_char_labels

print(predicted_char_labels[:5])  # shows first 5 predicted characters

cm = tf.math.confusion_matrix(labels=test_labels, predictions=y_predicted_labels)

# If cm is a TensorFlow tensor, convert to numpy
cm_np = cm.numpy() if hasattr(cm, 'numpy') else np.array(cm)

# Create long-form DataFrame
cm_long = pd.DataFrame(cm_np).stack().reset_index()
cm_long.columns = ['Truth', 'Predicted', 'Count']
cm_long = cm_long[cm_long['Count'] > 0]

# Map indices to characters for axis labels
class_labels = [label_map[i] for i in range(len(label_map))]
cm_long['Truth_char'] = cm_long['Truth'].map(lambda x: class_labels[x])
cm_long['Predicted_char'] = cm_long['Predicted'].map(lambda x: class_labels[x])

# Map character labels back to their indices for regression
char_to_index = {char: idx for idx, char in enumerate(class_labels)}
cm_long['Truth_idx'] = cm_long['Truth_char'].map(char_to_index)
cm_long['Predicted_idx'] = cm_long['Predicted_char'].map(char_to_index)

plt.figure(figsize=(12, 12))
# Scatter plot of the confusion matrix points
plt.scatter(
    cm_long['Predicted_idx'], cm_long['Truth_idx'],
    s=cm_long['Count'], c=cm_long['Count'], cmap='vlag', alpha=0.7, edgecolors='k'
)

# Dotted diagonal line from bottom left to top right
n_classes = len(class_labels)
plt.plot(
    [0, n_classes - 1],      # x: from left to right
    [0, n_classes - 1],      # y: from bottom to top (matches x)
    color='black', linewidth=2, linestyle='--'
)

# Set axis ticks and labels to characters
plt.xticks(ticks=range(len(class_labels)), labels=class_labels, rotation=0)
plt.yticks(ticks=range(len(class_labels)), labels=class_labels)
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.title('Confusion Matrix Scatterplot with Diagonal Reference Line')
plt.tight_layout()
plt.show()

plt.savefig('../../models/confusion_matrix_scatter.png', dpi=300, bbox_inches='tight')
plt.close()

model.save('../../models/chars_and_digits_model.h5')
