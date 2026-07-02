import base64
import logging
import os
import cv2
import numpy as np
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model

app = Flask(__name__)


class _RequestLogFilter(logging.Filter):
    """Suppress the per-request access logs while keeping Flask startup messages."""
    def filter(self, record):
        msg = record.getMessage()
        return not any(method in msg for method in ('POST /', 'GET /', 'HEAD /'))


logging.getLogger('werkzeug').addFilter(_RequestLogFilter())

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIGIT_PATH = os.path.join(BASE_DIR, 'models', 'digits_model.h5')
MODEL_CHAR_PATH = os.path.join(BASE_DIR, 'models', 'chars_and_digits_model.h5')
MAPPING_PATH = os.path.join(BASE_DIR, 'training', 'emnist-byclass-mapping.txt')

model_digit = load_model(MODEL_DIGIT_PATH)
model_char = load_model(MODEL_CHAR_PATH)


def load_emnist_mapping(mapping_file):
    mapping = {}
    with open(mapping_file, 'r') as f:
        for line in f:
            idx, unicode_int = line.strip().split()
            mapping[int(idx)] = chr(int(unicode_int))
    return mapping


emnist_mapping = load_emnist_mapping(MAPPING_PATH)


def preprocess_for_model(roi):
    """Original pipeline the model was trained on."""
    if len(roi.shape) == 3:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    else:
        gray = roi

    resized_high = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_CUBIC)
    blurred = cv2.GaussianBlur(resized_high, (3, 3), 0.5)
    resized = cv2.resize(blurred, (28, 28), interpolation=cv2.INTER_AREA)

    kernel = np.array([[-0.5, -0.5, -0.5],
                       [-0.5, 5.0, -0.5],
                       [-0.5, -0.5, -0.5]])
    sharpened = cv2.filter2D(resized, -1, kernel)

    thresh = cv2.adaptiveThreshold(
        sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 71, 15
    )
    normalized = thresh / 255.0
    return normalized


def preprocess_for_display(roi):
    """Crisp bilateral filter + Otsu for on-screen display only."""
    if len(roi.shape) == 3:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    else:
        gray = roi

    smoothed = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
    _, thresh = cv2.threshold(
        smoothed, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    return thresh


def predict(frame, mode):
    """Run inference on the central ROI of the supplied frame."""
    height, width = frame.shape[:2]

    roi_w = min(200, width)
    roi_h = min(200, height)
    x = (width - roi_w) // 2
    y = (height - roi_h) // 2
    roi = frame[y:y + roi_h, x:x + roi_w]

    normalized = preprocess_for_model(roi)
    display_thresh = preprocess_for_display(roi)

    _, thresh_buffer = cv2.imencode('.png', display_thresh)
    threshold_b64 = base64.b64encode(thresh_buffer).decode('utf-8')

    if mode == 'char':
        input_img = normalized.reshape(1, 28, 28, 1)
        prediction = model_char.predict(input_img, verbose=0)
        confidence = float(np.max(prediction))
        pred_class = int(np.argmax(prediction))
        pred_label = emnist_mapping.get(pred_class, '?')

        # Preserve the original script's 'a' preference logic.
        a_confidence = float(prediction[0][36])
        if a_confidence > 2.0:
            pred_class = 36
            pred_label = 'a'
            confidence = a_confidence

        confidence_threshold = 0.1
        mode_text = 'CHAR'
    else:
        input_img = normalized.reshape(1, 784)
        prediction = model_digit.predict(input_img, verbose=0)
        confidence = float(np.max(prediction))
        pred_class = int(np.argmax(prediction))
        pred_label = str(pred_class)
        confidence_threshold = 0.7
        mode_text = 'DIGIT'

    if confidence > confidence_threshold:
        label = f'{mode_text}: {pred_label} ({confidence:.2f})'
    else:
        label = f'No {mode_text.lower()} detected'

    return {
        'mode': mode,
        'prediction': pred_label,
        'confidence': confidence,
        'class': pred_class,
        'roi': {
            'x': x / width,
            'y': y / height,
            'w': roi_w / width,
            'h': roi_h / height,
        },
        'label': label,
        'threshold_image': threshold_b64,
    }


@app.route('/predict', methods=['POST'])
def predict_endpoint():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'Empty image file'}), 400

    mode = request.form.get('mode', 'digit').lower()
    if mode not in ('digit', 'char'):
        mode = 'digit'

    try:
        file_bytes = np.frombuffer(file.read(), np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({'error': 'Could not decode image'}), 400

        result = predict(frame, mode)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health_endpoint():
    return jsonify({'status': 'ok'})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, threaded=False)
