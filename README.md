# Live Number and Character Recognition

Real-time digit and character recognition using CNN models with webcam input.

## Overview

This project uses Convolutional Neural Networks (CNNs) to recognize handwritten digits (0-9) and characters (A-Z, a-z) in real-time through a webcam feed. The system includes two trained models:
- **Digits model** (`model.h5`): Recognizes numbers 0-9
- **Characters + Digits model** (`model_char_num.h5`): Recognizes both letters and numbers using EMNIST dataset

## Features

- Real-time webcam recognition with live preview
- Switchable modes between digit-only and character+digit recognition
- Preprocessing pipeline for improved accuracy (resizing, blur, sharpening, thresholding)
- Visual feedback with confidence scores
- ROI (Region of Interest) selection for focused recognition

## Live Demos

| Demo 1 | Demo 2 |
| --- | --- |
| ![Demo 1](Docs/char-demo-gif-1x-high-quality.gif) | ![Demo 2](Docs/char-and-num-demo-high-qual-1.25x.gif) |

## Model Insights

- **CharCNN confusion matrix:** ![Confusion matrix](Docs/CharCNN-Confusion-Matrix/output.png)
- **CNN activations before hidden layer:** ![CNN before hidden layer](Docs/CNN-B4-HIdden-Layer/Screenshot%202026-02-19%20145841.png)
- **CNN activations after hidden layer:** ![CNN after hidden layer](Docs/CNN-After-HIdden-Layer/Screenshot%202026-02-19%20150302.png)

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the webcam recognition:
```bash
python simple_webcam_recognition.py
```

## Usage

- **Q**: Quit the application
- **M**: Switch between digit mode and character mode
- Draw digits or characters in the ROI box (green rectangle)
- Recognition results display with confidence scores

## Training

### Digits-only model (`CNN.ipynb`)
1. **Download + preprocess MNIST:** Run the early cells to pull the dataset, visualize sample digits, normalize pixel values to `[0,1]`, and flatten images into 784-length vectors (`X_train_flattened`).
2. **Train baseline classifier:** Execute the dense-only model (single dense layer with sigmoid) for 5 epochs to learn a quick linear classifier and inspect the confusion matrix.
3. **Add hidden layer for higher accuracy:** Run the second model definition (ReLU hidden layer + sigmoid output) for ~10 epochs; this MLP captures nonlinear patterns and typically reaches ~97% accuracy.
4. **Evaluate + export:** Use `model.evaluate(X_test_flattened, y_test)` to verify accuracy, then persist weights via `model.save('model.h5')` for inference scripts like `simple_webcam_recognition.py`.

### Characters + digits model (`CharCNN.ipynb`)
1. **Pull the full EMNIST dataset:** Install `emnist` (see notebook comment) and run the download cell; this caches ~500 MB of `byclass` samples so the notebook can reuse them offline.
2. **Normalize + reshape for CNN input:** Convert grayscale images to `(batch, 28, 28, 1)` tensors and confirm the pixel range (min/max cell) so convolutional layers get standardized data.
3. **Build the CNN 2.0 stack:** The notebook constructs three Conv-BatchNorm blocks with pooling + dropout, followed by global average pooling and a dense layer outputting 62 logits—covering digits plus upper/lower-case letters.
4. **Train with smart callbacks:** Run `history = model.fit(...)` using `ReduceLROnPlateau` and `EarlyStopping` to automatically lower the learning rate and halt when validation loss plateaus (usually <20 epochs with 0.2 validation split).
5. **Evaluate, visualize, and save:** Evaluate against the EMNIST test set, run the confusion-matrix plotting cells for diagnostics, then export the final weights via `model.save('model_char_num.h5')`.

> Tip: `GPU_CHAR_CNN.ipynb` mirrors the char+digit workflow with GPU-specific tweaks if you have hardware acceleration available.

## Requirements

- Python 3.x
- OpenCV
- TensorFlow/Keras
- NumPy
- Matplotlib

