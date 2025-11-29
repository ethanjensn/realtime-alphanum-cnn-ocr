import cv2
import numpy as np
from tensorflow.keras.models import load_model

# .\.venv\Scripts\Activate.ps1
# python simple_webcam_recognition.py

# Load both models
model_digit = load_model('model.h5')  # Digits only
model_char = load_model('model_char_num.h5')  # Characters + digits

# Load EMNIST mapping for character recognition
def load_emnist_mapping(mapping_file):
    mapping = {}
    with open(mapping_file, 'r') as f:
        for line in f:
            idx, unicode_int = line.strip().split()
            mapping[int(idx)] = chr(int(unicode_int))
    return mapping

emnist_mapping = load_emnist_mapping('emnist-byclass-mapping.txt')

# Model selection: start with digits
use_char_model = False

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Define ROI (Region of Interest)
    x, y, w, h = 200, 100, 200, 200
    roi = frame[y:y+h, x:x+w]

    # Convert to grayscale
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # Enhanced preprocessing for better pixel density
    # Resize to higher resolution first for better detail
    resized_high = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_CUBIC)
    # Apply slight blur to reduce noise
    blurred = cv2.GaussianBlur(resized_high, (3, 3), 0.5)
    # Resize down to 28x28 with area interpolation
    resized = cv2.resize(blurred, (28, 28), interpolation=cv2.INTER_AREA)
    # Apply sharpening filter
    kernel = np.array([[-0.5,-0.5,-0.5], [-0.5,5,-0.5], [-0.5,-0.5,-0.5]])
    sharpened = cv2.filter2D(resized, -1, kernel)

    # Apply threshold to get black and white
    thresh = cv2.adaptiveThreshold(
        sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 71, 15
    )

    # Normalize for model input
    normalized = thresh / 255.0
    
    # Use different input format for each model
    if use_char_model:
        input_img = normalized.reshape(1, 28, 28, 1)  # CNN format
        model = model_char
    else:
        input_img = normalized.reshape(1, 784)  # Flattened format
        model = model_digit

    # Predict
    prediction = model.predict(input_img, verbose=0)
    confidence = np.max(prediction)
    pred_class = np.argmax(prediction)

    # Get prediction label
    if use_char_model:
        pred_label = emnist_mapping.get(pred_class, '?')
        confidence_threshold = 0.1  # Very low threshold for characters
        mode_text = "CHAR"
        
        # Debug: Show top 5 predictions for characters with mapping info
        top_indices = np.argsort(prediction[0])[-5:][::-1]
        print("Top 5 predictions:")
        for i, idx in enumerate(top_indices):
            char = emnist_mapping.get(idx, '?')
            conf = prediction[0][idx]
            print(f"  {i+1}. {char} (class {idx}, unicode {ord(char) if char != '?' else '?'}): {conf:.3f}")
        
        # Check if lowercase 'a' (class 36) is in top predictions
        a_class = 36
        a_confidence = prediction[0][a_class]
        print(f"Lowercase 'a' (class {a_class}): {a_confidence:.3f}")
        
        # Special handling: if 'a' confidence is reasonable, prefer it
        if a_confidence > 2.0:  # If 'a' has decent confidence
            pred_class = a_class
            pred_label = 'a'
            confidence = a_confidence
        
    else:
        pred_label = str(pred_class)
        confidence_threshold = 0.7
        mode_text = "DIGIT"

    # Display result
    if confidence > confidence_threshold:
        label = f'{mode_text}: {pred_label} ({confidence:.2f})'
        color = (0, 255, 0)  # Green
    else:
        label = f'No {mode_text.lower()} detected'
        color = (0, 0, 255)  # Red

    # Draw ROI and label
    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
    cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.putText(frame, f'Mode: {mode_text} (Press M to switch)', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Show frame and thresholded image
    cv2.imshow("Webcam", frame)
    cv2.imshow("Thresholded", cv2.resize(thresh, (200, 200)))

    # Handle key presses
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('m'):
        use_char_model = not use_char_model
        print(f"Switched to {'CHARACTER' if use_char_model else 'DIGIT'} mode")

cap.release()
cv2.destroyAllWindows() 