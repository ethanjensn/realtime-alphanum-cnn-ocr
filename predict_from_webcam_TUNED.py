import cv2
import numpy as np
from tensorflow.keras.models import load_model
from scipy.ndimage import center_of_mass
import tensorflow as tf

# .\.venv\Scripts\Activate.ps1
# python predict_from_webcam.py

# Load trained models
model_digit = load_model('model.h5')
model_char_num = load_model('model_char_num.h5')

# Load character mapping for EMNIST dataset
def load_emnist_mapping(mapping_file):
    mapping = {}
    with open(mapping_file, 'r') as f:
        for line in f:
            idx, unicode_int = line.strip().split()
            mapping[int(idx)] = chr(int(unicode_int))
    return mapping

emnist_mapping = load_emnist_mapping('emnist-byclass-mapping.txt')
use_char_num_model = False  # Start with digit-only model

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam. Exiting.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Define region of interest (ROI) for digit/character detection
    x, y, w, h = 200, 100, 200, 200
    roi = frame[y:y+h, x:x+w]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # Preprocess image based on model type
    if use_char_num_model:
        gray = cv2.GaussianBlur(gray, (3, 3), 0.3)
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_CUBIC)
        resized = cv2.resize(resized, (28, 28), interpolation=cv2.INTER_AREA)
        kernel = np.array([[-0.5,-0.5,-0.5], [-0.5,5,-0.5], [-0.5,-0.5,-0.5]])
        resized = cv2.filter2D(resized, -1, kernel)
    else:
        resized = cv2.resize(gray, (28, 28))

    # Apply thresholding to create binary image
    if use_char_num_model:
        _, thresh = cv2.threshold(resized, 127, 255, cv2.THRESH_BINARY_INV)
    else:
        thresh = cv2.adaptiveThreshold(resized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 27, 6)

    # Find contours to detect digit/character presence
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(cnt) for cnt in contours]

    # Check if digit/character is present based on contour area
    digit_present = False
    largest_cnt = None
    largest_area = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        min_area = 1 if use_char_num_model else 5
        max_area = 10000 if use_char_num_model else 5000
        if min_area < area < max_area:
            digit_present = True
            if area > largest_area:
                largest_area = area
                largest_cnt = cnt

    # Display debug windows
    roi_with_contours = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(roi_with_contours, contours, -1, (255, 0, 0), 2)
    thresh_large = cv2.resize(thresh, (420, 420), interpolation=cv2.INTER_NEAREST)
    cv2.imshow("Thresholded ROI", thresh_large)
    cv2.imshow("ROI with Contours", roi_with_contours)

    # Process detected digit/character
    if digit_present and largest_cnt is not None:
        x_cnt, y_cnt, w_cnt, h_cnt = cv2.boundingRect(largest_cnt)
        
        # Set cropping parameters based on model type
        if use_char_num_model:
            margin = 4
            min_size = 12
        else:
            margin = 1
            min_size = 16
            
        # Crop the digit/character with margin
        x_start = max(x_cnt - margin, 0)
        y_start = max(y_cnt - margin, 0)
        x_end = min(x_cnt + w_cnt + margin, thresh.shape[1])
        y_end = min(y_cnt + h_cnt + margin, thresh.shape[0])
        
        # Ensure minimum crop size
        crop_w = x_end - x_start
        crop_h = y_end - y_start
        if crop_w < min_size:
            extra = min_size - crop_w
            x_start = max(x_start - extra // 2, 0)
            x_end = min(x_end + (extra - extra // 2), thresh.shape[1])
        if crop_h < min_size:
            extra = min_size - crop_h
            y_start = max(y_start - extra // 2, 0)
            y_end = min(y_end + (extra - extra // 2), thresh.shape[0])
            
        digit_crop = thresh[y_start:y_end, x_start:x_end]
        
        # Handle empty crop error
        if digit_crop.size == 0:
            label = 'Error: Empty crop'
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
            cv2.imshow("Webcam", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('m'):
                use_char_num_model = not use_char_num_model
            continue

        # Resize and pad to 28x28 for model input
        if use_char_num_model:
            digit_resized = cv2.resize(digit_crop, (24, 24), interpolation=cv2.INTER_AREA)
            padded = np.pad(digit_resized, ((2,2),(2,2)), 'constant', constant_values=0)
        else:
            digit_resized = cv2.resize(digit_crop, (20, 20), interpolation=cv2.INTER_AREA)
            padded = np.pad(digit_resized, ((4,4),(4,4)), 'constant', constant_values=0)

        # Center the digit/character using center of mass
        padded_float = padded.astype(np.float32)
        cy_cx = center_of_mass(padded_float)
        if isinstance(cy_cx, tuple) and len(cy_cx) == 2:
            cy = float(cy_cx[0])
            cx = float(cy_cx[1])
        else:
            cy = 14.0
            cx = 14.0
        shiftx = int(np.round(14 - cx))
        shifty = int(np.round(14 - cy))
        M = np.array([[1, 0, shiftx], [0, 1, shifty]], dtype=np.float32)
        shifted = cv2.warpAffine(padded, M, (28, 28), borderValue=(0, 0, 0))

        # Normalize image for model input
        normalized = shifted / 255.0
        normalized = normalized.astype(np.float32)
        
        # Show processed image
        model_input_large = cv2.resize(shifted, (280, 280), interpolation=cv2.INTER_NEAREST)
        cv2.imshow("Model Input Digit", model_input_large)
        
        # Reshape for model input
        if use_char_num_model:
            input_img = normalized.reshape(1, 28, 28, 1)
        else:
            input_img = normalized.reshape(1, 784)
    else:
        # Fallback: use full ROI if no contour found
        normalized = thresh / 255.0
        normalized = normalized.astype(np.float32)
        
        if use_char_num_model:
            input_img = normalized.reshape(1, 28, 28, 1)
        else:
            input_img = normalized.reshape(1, 784)

    # Check image variance and set confidence thresholds
    roi_variance = np.var(normalized)
    variance_threshold = 0.005 if use_char_num_model else 0.01
    confidence_threshold = 0.4 if use_char_num_model else 0.7

    # Make prediction with error handling
    try:
        if use_char_num_model:
            prediction = model_char_num.predict(input_img, verbose=0)
            prediction = tf.nn.softmax(prediction, axis=-1)
        else:
            prediction = model_digit.predict(input_img, verbose=0)
    except Exception as e:
        label = f"Prediction Error: {str(e)[:20]}..."
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2, cv2.LINE_AA)
        cv2.imshow("Webcam", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('m'):
            use_char_num_model = not use_char_num_model
        continue

    # Display prediction results
    confidence = np.max(prediction)
    
    if digit_present and confidence > confidence_threshold and roi_variance > variance_threshold:
        pred_class = np.argmax(prediction)
        if use_char_num_model:
            pred_label = emnist_mapping.get(pred_class, '?')
        else:
            pred_label = str(pred_class)
        label = f'Prediction: {pred_label} ({confidence:.2f})'
    else:
        label = 'No character detected' if use_char_num_model else 'No digit detected'

    # Draw results on frame
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)

    # Display frame and handle key presses
    cv2.imshow("Webcam", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('m'):
        use_char_num_model = not use_char_num_model

print("Exiting program. Releasing webcam and destroying windows.")

cap.release()
cv2.destroyAllWindows()