"""Convert Keras .h5 models to TensorFlow Lite .tflite for on-device Android inference.

The digit model originally accepts flattened (1, 784) input. We wrap it in a new
Sequential model with Input(28, 28, 1) -> Flatten() -> original layers so both
models share the same (28, 28, 1) input shape.

Usage:
    cd ocr-desktop
    python convert_to_tflite.py
"""

import os
import numpy as np
import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIGIT_PATH = os.path.join(BASE_DIR, "models", "digits_model.h5")
MODEL_CHAR_PATH = os.path.join(BASE_DIR, "models", "chars_and_digits_model.h5")
TFLITE_DIGIT_PATH = os.path.join(BASE_DIR, "models", "digits_model.tflite")
TFLITE_CHAR_PATH = os.path.join(BASE_DIR, "models", "chars_and_digits_model.tflite")


def convert_digit_model():
    """Load the digit model and re-export with (28, 28, 1) input, then convert to TFLite."""
    print("Loading digit model...")
    original = tf.keras.models.load_model(MODEL_DIGIT_PATH)

    # Build a new model that accepts (28, 28, 1) and flattens internally
    new_model = tf.keras.Sequential()
    new_model.add(tf.keras.layers.Input(shape=(28, 28, 1)))
    new_model.add(tf.keras.layers.Flatten())

    # Copy all layers from the original model (skip its input layer if any)
    for layer in original.layers:
        new_model.add(layer)

    # Copy weights from original to new model
    # The original model's layers should map directly since we just added a Flatten
    for i, layer in enumerate(original.layers):
        new_model.layers[i + 1].set_weights(layer.get_weights())

    # Verify the new model works
    test_input = np.random.rand(1, 28, 28, 1).astype(np.float32)
    original_input = test_input.reshape(1, 784)
    orig_pred = original.predict(original_input, verbose=0)
    new_pred = new_model.predict(test_input, verbose=0)
    print(f"  Original output shape: {orig_pred.shape}")
    print(f"  New model output shape: {new_pred.shape}")
    print(f"  Outputs match: {np.allclose(orig_pred, new_pred, atol=1e-5)}")

    print("Converting digit model to TFLite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(new_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    with open(TFLITE_DIGIT_PATH, "wb") as f:
        f.write(tflite_model)
    print(f"  Saved: {TFLITE_DIGIT_PATH} ({len(tflite_model) / 1024:.1f} KB)")


def convert_char_model():
    """Convert the char/digit model to TFLite (already has (28, 28, 1) input)."""
    print("Loading char+digits model...")
    model = tf.keras.models.load_model(MODEL_CHAR_PATH)

    print("Converting char+digits model to TFLite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    with open(TFLITE_CHAR_PATH, "wb") as f:
        f.write(tflite_model)
    print(f"  Saved: {TFLITE_CHAR_PATH} ({len(tflite_model) / 1024:.1f} KB)")


if __name__ == "__main__":
    print(f"TensorFlow version: {tf.__version__}")
    convert_digit_model()
    convert_char_model()
    print("\nDone! Copy the .tflite files to ocr-android/app/src/main/assets/")
