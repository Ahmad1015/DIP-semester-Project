import cv2
import numpy as np
import pytesseract
import streamlit as st
from PIL import Image
from ultralytics import YOLO

def ensure_uint8(image):
    """Ensure image is uint8 type and in correct range."""
    if isinstance(image, tuple):
        image = image[0]
    if image.dtype != np.uint8:
        image = (image * 255).clip(0, 255).astype(np.uint8)
    return image

def get_yolo_detections(image):
    """Detect license plate using YOLO and return bounding boxes."""
    model = YOLO('best.pt')
    results = model(image)
    boxes = []
    for result in results:
        for detection in result.boxes:
            x1, y1, x2, y2 = detection.xyxy[0]
            boxes.append((int(x1), int(y1), int(x2), int(y2)))
    return boxes

def binary_and_contour_crop(image):
    """Convert image to binary and crop around character contours."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    height, width = image.shape[:2]
    candidate_contours = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / float(h)
        if 0.2 < aspect_ratio < 1.0 and h > height * 0.3:  # Likely a character contour
            candidate_contours.append(contour)

    if candidate_contours:
        x_min = min(cv2.boundingRect(c)[0] for c in candidate_contours)
        y_min = min(cv2.boundingRect(c)[1] for c in candidate_contours)
        x_max = max(cv2.boundingRect(c)[0] + cv2.boundingRect(c)[2] for c in candidate_contours)
        y_max = max(cv2.boundingRect(c)[1] + cv2.boundingRect(c)[3] for c in candidate_contours)
        return binary[y_min:y_max, x_min:x_max]
    else:
        return binary  # Return full binary image if no contours found

def perform_ocr(image, config=r'--oem 1 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'):
    """Perform OCR on the given image."""
    return pytesseract.image_to_string(image, config=config).strip()

def image_blurred(image):
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    
    # Apply the filter to the image
    sharpened_image = cv2.filter2D(image, -1, kernel)
    
    return sharpened_image
def detect_salt_pepper_noise(image, threshold=0.01):
    """
    Detect the presence of salt-and-pepper noise in an image by analyzing pixel intensities.
    
    Parameters:
    - image (numpy.ndarray): The input image.
    - threshold (float): Proportion of pixels at 0 or 255 to consider noise.
    
    Returns:
    - bool: True if noise is detected, otherwise False.
    """
    total_pixels = image.size
    black_pixels = np.sum(image == 0)
    white_pixels = np.sum(image == 255)
    
    # Calculate the proportion of extreme pixels
    noise_ratio = (black_pixels + white_pixels) / total_pixels
    return noise_ratio > threshold


st.title("License Plate Detection with OCR")

uploaded_image = st.file_uploader("Upload an image for OCR", type=["jpg", "jpeg", "png"])

if uploaded_image:
    # Convert the uploaded file to an image
    image = Image.open(uploaded_image)
    image = np.array(image)
    image = ensure_uint8(image)

    st.image(image, caption="Uploaded Image", use_container_width=True)

    laplacian_var = cv2.Laplacian(image, cv2.CV_64F).var()
    if laplacian_var < 100:
        st.warning("The uploaded image may be blurry. Please upload a clearer image for better results.")
        st.warning("Trying to do OCR on Blurr Image")
        image = image_blurred(image)
        st.image(image, caption="After Applying Sharpening Filter", use_container_width=True)
    if detect_salt_pepper_noise(image):
        image = cv2.medianBlur(image, 3)
        st.warning("The uploaded image may contain salt-and-pepper noise. Applying median filter to remove noise.")
        st.image(image, caption="After Applying Median Filter with window size 3", use_container_width=True)

    # Convert RGBA to RGB if needed
    if image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

    # Get YOLO detections and use the first bounding box if available
    boxes = get_yolo_detections(image)
    if boxes:
        bbox = boxes[0]
        x1, y1, x2, y2 = bbox
        model_cropped_image = image[y1:y2, x1:x2]
        st.image(model_cropped_image, caption="Cropping the Bounding Boxes returned by the Model", use_container_width=True)

        # Further process the model-cropped image to isolate central characters
        refined_cropped_plate = binary_and_contour_crop(model_cropped_image)
        st.image(refined_cropped_plate, caption="Binarization of our Image", use_container_width=True)

        # Perform OCR on the refined cropped image
        ocr_result = perform_ocr(refined_cropped_plate)
        st.write("OCR Result:")
        st.write(f"Text: {ocr_result}")
    else:
        # Fallback to OCR on the entire image if no bounding box is detected
        st.write("No bounding box detected. Performing OCR on the entire image as a potential license plate.")

        refined_cropped_plate = binary_and_contour_crop(image)
        st.image(refined_cropped_plate, caption="Refined Cropped Black-and-White License Plate (Fallback)", use_container_width=True)

        # Perform OCR on the fallback refined cropped image
        ocr_result = perform_ocr(refined_cropped_plate)
        st.write("Potential License Plate with OCR Result:")
        st.write(f"Text: {ocr_result}")
else:
    st.write("Please upload an image to proceed.")
