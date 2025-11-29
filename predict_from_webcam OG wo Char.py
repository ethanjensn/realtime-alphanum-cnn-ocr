import cv2
import numpy as np
from tensorflow.keras.models import load_model
from scipy.ndimage import center_of_mass

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
    #The first part (y:y+h) is for rows (vertical slice).
    # The second part (x:x+w) is for columns (horizontal slice).

    # Preprocess ROI for model
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) #turns it into a grayscale image
    resized = cv2.resize(gray, (28, 28)) #resizes the image to 28x28 pixels

    # Use the thresholded image for prediction, ensuring white digit on black background
    thresh = cv2.adaptiveThreshold(
        resized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 27, 6
        # Gaussian It calculates a weighted average of all the pixel values in that region.
        # The weights are determined by a Gaussian function: pixels closer to the center have more influence, 
        # those farther away have less.
        # this is better than a simple mean

        # Binary Inverse:
        # This means that the pixels that are black (0) will become white (255), and the pixels that are white (255) will become black (0).
        # This is useful because it makes the digits easier to detect.

        # 27 is max block size
        # 27, 6 is good
        # C is 4
            # Lower values (e.g., 0, 2) make it easier for pixels to become white (less dense, more noise).
            # Higher values (e.g., 6, 8, 10) make it harder for pixels to become white (more dense, less noise).
    )

    # Digit presence check using adaptive thresholding and contours (use resized for contours)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Contours are the boundaries or outlines of white shapes
    # findContours is used to detect the outlines of white shapes
    # Thresh is the binary image

    # RETR_EXTERNAL: ignores any contours that are inside other contours (like holes or islands inside the digit).
    # If you draw a “0”, which is a loop, RETR_EXTERNAL will only find the outer boundary, not the inner hole.

    # CHAIN_APPROX_SIMPLE:
    # This is a faster way to find the contours.
    # It only stores the endpoints of the contours, not the entire contour.
    # This is useful because it reduces the amount of memory needed to store the contours.

    areas = [cv2.contourArea(cnt) for cnt in contours]
    # contourArea is used to calculate the area of the contours
    # cnt is the contour
    # areas is a list of the areas of the contours

    print("Contour areas:", areas)
    # prints those sizes, so you can see what the code is detecting in real time.

    # the 2 line of code above are necessary to detect the digit, they can be removed
    
    digit_present = False
    largest_cnt = None
    largest_area = 0
    for cnt in contours: # Go through each detected contour (shape) in the image.
        area = cv2.contourArea(cnt) # Compute the area (number of pixels inside) for the current contour.
        if 5 < area < 5000:  # Loosened area range for digit detection
            digit_present = True # If the area is between 5 and 5000 pixels, then the digit is present.
            if area > largest_area: # If the area is larger than the largest area found so far, then update the largest area and the largest contour.
                largest_area = area
                largest_cnt = cnt
    # remeber A contour is simply the outline or boundary of a shape in an image.


    # Show thresholded ROI and contours for debugging
    roi_with_contours = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(roi_with_contours, contours, -1, (255, 0, 0), 2)
    # Enlarge the thresholded ROI for display
    thresh_large = cv2.resize(thresh, (420, 420), interpolation=cv2.INTER_NEAREST)
    cv2.imshow("Thresholded ROI", thresh_large)
    cv2.imshow("ROI with Contours", roi_with_contours)

    # For model input: auto-crop, resize, and pad the digit to MNIST style if a contour is found
    if digit_present and largest_cnt is not None:
        x_cnt, y_cnt, w_cnt, h_cnt = cv2.boundingRect(largest_cnt)
        margin = 1
        x_start = max(x_cnt - margin, 0)
        y_start = max(y_cnt - margin, 0)
        x_end = min(x_cnt + w_cnt + margin, thresh.shape[1])
        y_end = min(y_cnt + h_cnt + margin, thresh.shape[0])
        # Enforce minimum crop size
        min_size = 16
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
        # Resize to 20x20
        digit_resized = cv2.resize(digit_crop, (20, 20), interpolation=cv2.INTER_AREA)
        # Pad to 28x28
        padded = np.pad(digit_resized, ((4,4),(4,4)), 'constant', constant_values=0)

        # Center the digit by its center of mass
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

        normalized = shifted / 255.0
        input_img = normalized.reshape(1, 784)
        # Show the actual digit image being sent to the model
        model_input_large = cv2.resize(shifted, (280, 280), interpolation=cv2.INTER_NEAREST)
        cv2.imshow("Model Input Digit", model_input_large)
    else:
        # Fallback to using the full thresholded ROI
        normalized = thresh / 255.0
        input_img = normalized.reshape(1, 784)

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
