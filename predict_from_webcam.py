import cv2
import numpy as np
from tensorflow.keras.models import load_model
from scipy.ndimage import center_of_mass
import os

# .\.venv\Scripts\Activate.ps1
# python predict_from_webcam.py

# Load both models
model_digit = load_model('model.h5')  # Digits only
model_char_num = load_model('model_char_num.h5')  # Digits + chars

# Load EMNIST mapping for char+digit model
def load_emnist_mapping(mapping_file):
    mapping = {}
    with open(mapping_file, 'r') as f:
        for line in f:
            idx, unicode_int = line.strip().split()
            mapping[int(idx)] = chr(int(unicode_int))
    return mapping

emnist_mapping = load_emnist_mapping('emnist-byclass-mapping.txt')

# Model selection: start with digit-only, press 'm' to toggle
use_char_num_model = False

# Open the default webcam (0)
cap = cv2.VideoCapture(1)

# Check if webcam opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam. Exiting.")
    exit()

print("Webcam opened successfully. Starting main loop...") # Debug Print - Initial

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame - breaking loop") # Debug Print 1
        break

    print("Successfully grabbed frame.") # Debug Print 1 continued

    # Define ROI (Region of Interest) - adjust as needed
    x, y, w, h = 200, 100, 200, 200  # Back to original 200x200 size
    roi = frame[y:y+h, x:x+w]
    #The first part (y:y+h) is for rows (vertical slice).
    # The second part (x:x+w) is for columns (horizontal slice).

    # Preprocess ROI for model
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) #turns it into a grayscale image
    
    # Use different preprocessing for characters vs digits
    if use_char_num_model:
        # Moderate high-quality preprocessing for characters
        # Apply very slight Gaussian blur to reduce noise
        gray = cv2.GaussianBlur(gray, (3, 3), 0.3)
        
        # Use moderate resolution resize for better pixel density
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_CUBIC) #resizes to 84x84 for better detail
        # Then resize down to 28x28 for model input
        resized = cv2.resize(resized, (28, 28), interpolation=cv2.INTER_AREA) #final resize to 28x28
        
        # Apply very light sharpening to enhance edges
        kernel = np.array([[-0.5,-0.5,-0.5], [-0.5,5,-0.5], [-0.5,-0.5,-0.5]])
        resized = cv2.filter2D(resized, -1, kernel)
    else:
        # Original simple preprocessing for digits (exactly as it was before any changes)
        resized = cv2.resize(gray, (28, 28)) #direct resize to 28x28

    print(f"ROI and resize done. Resized shape: {resized.shape}, Min/Max: {resized.min()}/{resized.max()}") # Debug Print 2

    # Use different preprocessing for character vs digit models
    if use_char_num_model:
        # For characters: use simple thresholding similar to EMNIST preprocessing
        _, thresh = cv2.threshold(resized, 127, 255, cv2.THRESH_BINARY_INV)  # Back to original for better detail
        print("Character mode: Simple threshold applied.") # Debug Print 3a
    else:
        # For digits: use adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            resized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 27, 6  # Back to original for better detail
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
        print("Digit mode: Adaptive threshold applied.") # Debug Print 3b

    print(f"Threshold applied. Thresh shape: {thresh.shape}, Dtype: {thresh.dtype}, Min/Max: {thresh.min()}/{thresh.max()}") # Debug Print 3c
    
    # Optional: Add morphological operations to make characters thicker
    # Uncomment the lines below if you want even thicker characters
    # kernel = np.ones((2,2), np.uint8)  # 2x2 kernel for dilation
    # thresh = cv2.dilate(thresh, kernel, iterations=1)  # Makes white areas larger
    # print("Applied dilation to make characters thicker.")

    # Digit presence check using adaptive thresholding and contours (use resized for contours)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Contours are the boundaries or outlines of white shapes
    # findContours is used to detect the outlines of white shapes
    # Thresh is the binary image

    # RETR_EXTERNAL: ignores any contours that are inside other contours (like holes or islands inside the digit).
    # If you draw a "0", which is a loop, RETR_EXTERNAL will only find the outer boundary, not the inner hole.

    # CHAIN_APPROX_SIMPLE:
    # This is a faster way to find the contours.
    # It only stores the endpoints of the contours, not the entire contour.
    # This is useful because it reduces the amount of memory needed to store the contours.

    areas = [cv2.contourArea(cnt) for cnt in contours]
    # contourArea is used to calculate the area of the contours
    # cnt is the contour
    # areas is a list of the areas of the contours

    print("Contour areas:", areas) # Debug Print - original
    print(f"Contours found: {len(contours)}") # Debug Print 4
    # prints those sizes, so you can see what the code is detecting in real time.

    # the 2 line of code above are necessary to detect the digit, they can be removed
    
    digit_present = False
    largest_cnt = None
    largest_area = 0
    for cnt in contours: # Go through each detected contour (shape) in the image.
        area = cv2.contourArea(cnt) # Compute the area (number of pixels inside) for the current contour.
        # Adjust area range based on model type - characters might be smaller
        min_area = 1 if use_char_num_model else 5  # Very low minimum for characters
        max_area = 10000 if use_char_num_model else 5000  # Higher maximum for characters
        if min_area < area < max_area:  # Adjusted area range for both digits and characters
            digit_present = True # If the area is within range, then a character/digit is present.
            if area > largest_area: # If the area is larger than the largest area found so far, then update the largest area and the largest contour.
                largest_area = area
                largest_cnt = cnt
    # remember A contour is simply the outline or boundary of a shape in the image.


    # Show thresholded ROI and contours for debugging
    roi_with_contours = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(roi_with_contours, contours, -1, (255, 0, 0), 2)
    # Enlarge the thresholded ROI for display
    thresh_large = cv2.resize(thresh, (420, 420), interpolation=cv2.INTER_NEAREST)
    cv2.imshow("Thresholded ROI", thresh_large)
    cv2.imshow("ROI with Contours", roi_with_contours)
    print("OpenCV windows 'Thresholded ROI' and 'ROI with Contours' displayed.") # Debug Print 5
    
    # Debug: Show preprocessing info
    if use_char_num_model:
        print(f"Character mode - Contours found: {len(contours)}, Areas: {areas}")
        print(f"Threshold type: {'Simple' if use_char_num_model else 'Adaptive'}")

    # For model input: auto-crop, resize, and pad the digit to MNIST style if a contour is found
    if digit_present and largest_cnt is not None:
        x_cnt, y_cnt, w_cnt, h_cnt = cv2.boundingRect(largest_cnt)
        
        print(f"Largest contour bounds: x:{x_cnt}, y:{y_cnt}, w:{w_cnt}, h:{h_cnt}") # Debug Print - contour bounds
        
        # Use different margins and cropping for characters vs digits
        if use_char_num_model:
            margin = 4  # Larger margin for characters to avoid cutting off parts
            min_size = 12  # Smaller minimum size for characters
        else:
            margin = 1  # Smaller margin for digits
            min_size = 16  # Larger minimum size for digits
            
        x_start = max(x_cnt - margin, 0)
        y_start = max(y_cnt - margin, 0)
        x_end = min(x_cnt + w_cnt + margin, thresh.shape[1])
        y_end = min(y_cnt + h_cnt + margin, thresh.shape[0])
        
        # Enforce minimum crop size
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
        
        # Check if digit_crop is empty, can happen if bounds are invalid
        if digit_crop.size == 0:
            print("Warning: digit_crop is empty. Skipping prediction for this frame.")
            label = 'Error: Empty crop'
            # Continue the loop to process the next frame
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2) # Red rectangle for error
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
            cv2.imshow("Webcam", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('m'):
                use_char_num_model = not use_char_num_model
                print(f"Switched to {'Char+Num' if use_char_num_model else 'Digit-only'} model.")
            continue # Skip to the next frame

        # Use different resize dimensions for characters vs digits
        if use_char_num_model:
            # For characters: resize to 24x24 then pad to 28x28 (more space for character details)
            digit_resized = cv2.resize(digit_crop, (24, 24), interpolation=cv2.INTER_AREA)
            padded = np.pad(digit_resized, ((2,2),(2,2)), 'constant', constant_values=0)
        else:
            # For digits: resize to 20x20 then pad to 28x28
            digit_resized = cv2.resize(digit_crop, (20, 20), interpolation=cv2.INTER_AREA)
            padded = np.pad(digit_resized, ((4,4),(4,4)), 'constant', constant_values=0)

        print(f"Digit crop processed. Resized shape: {digit_resized.shape}, Padded shape: {padded.shape}") # Debug Print - padded shape

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
        normalized = normalized.astype(np.float32) # Ensure float32

        print(f"Image normalized and shifted. Shape: {normalized.shape}, Dtype: {normalized.dtype}, Min/Max: {normalized.min():.2f}/{normalized.max():.2f}") # Debug Print 6
        
        # Show the actual digit image being sent to the model
        model_input_large = cv2.resize(shifted, (280, 280), interpolation=cv2.INTER_NEAREST)
        cv2.imshow("Model Input Digit", model_input_large) # Debug Print 8
        
        if use_char_num_model:
            # For CNNs, reshape to (batch_size, height, width, channels)
            input_img = normalized.reshape(1, 28, 28, 1)
            print(f"Character model input_img reshaped to: {input_img.shape}") # Debug Print 7a
        else:
            # For the digit model (assuming it's a flattened input model)
            input_img = normalized.reshape(1, 784)
            print(f"Digit model input_img reshaped to: {input_img.shape}") # Debug Print 7b
        
        # Debug: Show input statistics and crop info
        if use_char_num_model:
            print(f"Input stats - min: {normalized.min():.3f}, max: {normalized.max():.3f}, mean: {normalized.mean():.3f}")
            print(f"Crop info - Original: {digit_crop.shape}, Resized: {digit_resized.shape}, Final: {padded.shape}")
            print(f"Crop bounds - x:{x_start}-{x_end}, y:{y_start}-{y_end}, margin: {margin}")
    else:
        # Fallback to using the full thresholded ROI if no valid contour found
        normalized = thresh / 255.0
        normalized = normalized.astype(np.float32) # Ensure float32
        
        if use_char_num_model:
            input_img = normalized.reshape(1, 28, 28, 1)
            print(f"Fallback Character model input_img reshaped to: {input_img.shape}")
        else:
            input_img = normalized.reshape(1, 784)
            print(f"Fallback Digit model input_img reshaped to: {input_img.shape}")

    # Additional check: Only predict if ROI has enough variance (i.e., not just blank/background)
    roi_variance = np.var(normalized)
    # Lower variance threshold for characters - they might have less variance than digits
    variance_threshold = 0.005 if use_char_num_model else 0.01

    # More reasonable confidence thresholds
    confidence_threshold = 0.4 if use_char_num_model else 0.7

    # Predict digit/char only if ROI is not blank, model is confident, and character-like contour is present
    try:
        print("Attempting model prediction...") # Debug Print 9
        if use_char_num_model:
            prediction = model_char_num.predict(input_img, verbose=0)
            # Apply softmax to convert logits to probabilities
            import tensorflow as tf
            prediction = tf.nn.softmax(prediction, axis=-1)
            print(f"Applied softmax to character predictions. Max value: {np.max(prediction):.3f}")
        else:
            prediction = model_digit.predict(input_img, verbose=0)
        print("Prediction successful.") # Debug Print 10
    except Exception as e:
        print(f"****** CRITICAL ERROR DURING PREDICTION: {e} ******")
        print(f"Input shape at crash: {input_img.shape}")
        print(f"Input dtype at crash: {input_img.dtype}")
        print(f"Model type: {'char+digit' if use_char_num_model else 'digit-only'}")
        
        # Try to get input shape from the loaded models directly
        try:
            print(f"Char model expected input shape: {model_char_num.input_shape}")
        except Exception:
            print("Could not get model_char_num.input_shape")
        try:
            print(f"Digit model expected input shape: {model_digit.input_shape}")
        except Exception:
            print("Could not get model_digit.input_shape")
            
        label = f"Prediction Error: {str(e)[:20]}..." # Display short error message on screen
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2, cv2.LINE_AA)
        cv2.imshow("Webcam", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('m'):
            use_char_num_model = not use_char_num_model
            print(f"Switched to {'Char+Num' if use_char_num_model else 'Digit-only'} model.")
        continue # Skip to the next frame in case of prediction error
        # If you want the program to fully stop on error, uncomment 'raise' below:
        # raise # Uncomment this to get a full Python traceback and stop execution

    confidence = np.max(prediction)
    
    if digit_present and confidence > confidence_threshold and roi_variance > variance_threshold:
        pred_class = np.argmax(prediction)
        if use_char_num_model:
            pred_label = emnist_mapping.get(pred_class, '?')
            print(f"Character prediction: {pred_label} (class {pred_class}) with confidence {confidence:.3f}")
        else:
            pred_label = str(pred_class)
            print(f"Digit prediction: {pred_label} with confidence {confidence:.3f}")
        label = f'Prediction: {pred_label} ({confidence:.2f})'
    else:
        label = 'No character detected' if use_char_num_model else 'No digit detected'
        if digit_present:
            print(f"No prediction - confidence: {confidence:.3f}, variance: {roi_variance:.3f}, threshold: {confidence_threshold:.3f}")
            # Show top 3 predictions for debugging
            if use_char_num_model:
                top_indices = np.argsort(prediction[0])[-3:][::-1]
                print("Top 3 predictions:")
                for i, idx in enumerate(top_indices):
                    char = emnist_mapping.get(idx, '?')
                    conf = prediction[0][idx]
                    print(f"  {i+1}. {char} (class {idx}): {conf:.3f}")

    # Draw ROI rectangle and prediction
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.putText(frame, label, (x, y-10),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)

    # Show the frame
    cv2.imshow("Webcam", frame)
    key = cv2.waitKey(1) & 0xFF # Debug Print 12: Ensure waitKey is called
    if key == ord('q'):
        print(" 'q' pressed. Exiting.") # Debug Print - exit on q
        break
    elif key == ord('m'):
        use_char_num_model = not use_char_num_model
        print(f"Switched to {'Char+Num' if use_char_num_model else 'Digit-only'} model.") # Debug Print - model switch

print("Exiting program. Releasing webcam and destroying windows.") # Debug Print 13

cap.release()
cv2.destroyAllWindows()