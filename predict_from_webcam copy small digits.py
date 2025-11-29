import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load your trained model
model = load_model('model.h5')  # Update with your model's filename

# Open the default webcam (0)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Define ROI (Region of Interest) - adjust as needed
    x, y, w, h = 200, 100, 200, 200
    roi = frame[y:y+h, x:x+w]

    # Preprocess ROI for model
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (28, 28))

    # Use the thresholded image for prediction, ensuring white digit on black background
    thresh = cv2.adaptiveThreshold(
        resized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 109, 6
        # 159, 6
        # 89, 6
        # block size is 79, 
        # C is 4
            # Lower values (e.g., 0, 2) make it easier for pixels to become white (less dense, more noise).
            # Higher values (e.g., 6, 8, 10) make it harder for pixels to become white (more dense, less noise).
    )

    # Digit presence check using adaptive thresholding and contours (use resized for contours)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(cnt) for cnt in contours]
    print("Contour areas:", areas)
    digit_present = False
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 20 < area < 2000:  # Loosened area range for digit detection
            digit_present = True
            break

    # Show thresholded ROI and contours for debugging
    roi_with_contours = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(roi_with_contours, contours, -1, (255, 0, 0), 2)
    # Enlarge the thresholded ROI for display
    thresh_large = cv2.resize(thresh, (420, 420), interpolation=cv2.INTER_NEAREST)
    cv2.imshow("Thresholded ROI", thresh_large)
    cv2.imshow("ROI with Contours", roi_with_contours)

    # Use the thresholded, resized (28x28), white-on-black ROI for prediction
    normalized = thresh / 255.0
    input_img = normalized.reshape(1, 28, 28, 1)

    # Additional check: Only predict if ROI has enough variance (i.e., not just blank/background)
    roi_variance = np.var(normalized)
    variance_threshold = 0.01  # Adjust as needed; higher = more strict

    # Predict digit only if ROI is not blank, model is confident, and digit-like contour is present
    prediction = model.predict(input_img)
    confidence = np.max(prediction)
    if digit_present and confidence > 0.8 and roi_variance > variance_threshold:
        digit = np.argmax(prediction)
        label = f'Prediction: {digit} ({confidence:.2f})'
    else:
        label = 'No digit detected'

    # Draw ROI rectangle and prediction
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.putText(frame, label, (x, y-10),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)

    # Show the frame
    cv2.imshow("Webcam", frame)

    # Break on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
