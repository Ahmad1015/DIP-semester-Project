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

st.title("License Plate Detection with Model-Based Cropping")

uploaded_image = st.file_uploader("Upload an image for OCR", type=["jpg", "jpeg", "png"])

if uploaded_image:
    # Convert the uploaded file to an image
    image = Image.open(uploaded_image)
    image = np.array(image)
    image = ensure_uint8(image)

    # Convert RGBA to RGB if needed
    if image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Get YOLO detections and use the first bounding box if available
    boxes = get_yolo_detections(image)
    if boxes:
        bbox = boxes[0]
        x1, y1, x2, y2 = bbox
        model_cropped_image = image[y1:y2, x1:x2]
        st.image(model_cropped_image, caption="Model-Based Cropped License Plate", use_container_width=True)

        # Further process the model-cropped image to isolate central characters
        refined_cropped_plate = binary_and_contour_crop(model_cropped_image)
        st.image(refined_cropped_plate, caption="Refined Cropped Black-and-White License Plate", use_container_width=True)

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
