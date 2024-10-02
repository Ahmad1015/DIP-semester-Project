import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import pytesseract

# 1. Sharpening
def sharpen_image(image):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])  # Sharpening kernel
    sharpened = cv2.filter2D(src=image, ddepth=-1, kernel=kernel)
    return sharpened

# 2. CLAHE (Histogram Equalization)
def apply_clahe(image):
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    lab = cv2.merge((cl, a, b))
    equalized_image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return equalized_image

# 3. Bilateral Filtering
def apply_bilateral_filter(image):
    bilateral_filtered = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)
    return bilateral_filtered

# 4. Gamma Correction
def adjust_gamma(image, gamma=1.0):
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    gamma_corrected = cv2.LUT(image, table)
    return gamma_corrected

# 5. Image Denoising
def denoise_image(image):
    denoised = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
    return denoised

# 6. Edge Detection Enhancement
def enhance_edges(image):
    edges = cv2.Canny(image, 100, 200)
    return edges

# 7. Brightness and Contrast Adjustment
def adjust_brightness_contrast(image, brightness=30, contrast=30):
    adjusted = cv2.convertScaleAbs(image, alpha=1 + (contrast / 127.), beta=brightness)
    return adjusted

 # Function to draw red bounding boxes and crop license plates
def draw_red_bboxes_and_crop(results, image, thickness=4):
    cropped_plates = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), thickness)
            cropped_plate = image[y1:y2, x1:x2]
            cropped_plates.append(cropped_plate)
    return image, cropped_plates

# Function to perform YOLOv8 inference, annotate the image, and crop license plates
def infer_annotate_and_crop(image):
    results = model(image)
    annotated_image, cropped_plates = draw_red_bboxes_and_crop(results, image.copy())
    return annotated_image, cropped_plates

# Function to perform OCR on cropped license plates
def perform_ocr(cropped_plates):
    ocr_results = []
    for idx, plate in enumerate(cropped_plates):
        # Convert to grayscale
        gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
        # Apply thresholding to preprocess the image
        threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        # Perform OCR on the thresholded image
        text = pytesseract.image_to_string(threshold, config='--psm 8 --oem 3')
        ocr_results.append(text.strip())
    return ocr_results

model = YOLO("best.pt")


def main():
    st.title("Digital Image Processing Semester Project - License Plate Detection and OCR")

    # File uploader for image
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Read the image file
        image = Image.open(uploaded_file)
        image = np.array(image)

        # Apply digital image processing steps
        st.write("Processing the image...")
        processed_image = sharpen_image(image)
        processed_image = apply_clahe(processed_image)
        processed_image = apply_bilateral_filter(processed_image)
        processed_image = adjust_gamma(processed_image, gamma=1.2)
        processed_image = denoise_image(processed_image)
        processed_image = adjust_brightness_contrast(processed_image)

        # Perform YOLOv8 inference, get annotated image with red bounding boxes, and crop license plates
        annotated_image, cropped_plates = infer_annotate_and_crop(processed_image.copy())

        # Perform OCR on cropped license plates
        ocr_results = perform_ocr(cropped_plates)

        # Display the processed image (left) and the YOLOv8 image with red bounding boxes (right)
        col1, col2 = st.columns(2)

        with col1:
            st.image(processed_image, caption="Processed Image", use_column_width=True)

        with col2:
            st.image(annotated_image, caption="License Plate Detection with Red Bounding Boxes", use_column_width=True)

        # Display OCR results using Streamlit components
        st.subheader("OCR Results")
        
        if ocr_results:
            for idx, result in enumerate(ocr_results, start=1):
                with st.expander(f"Plate", expanded=True):
                    st.text_input("OCR Result", value=result, key=f"plate_{idx}")
        else:
            st.warning("No license plates detected or OCR failed to recognize text.")

        # Display cropped license plates
        st.subheader("Cropped License Plates")
        cropped_cols = st.columns(len(cropped_plates))
        for idx, (plate, col) in enumerate(zip(cropped_plates, cropped_cols)):
            col.image(plate, caption=f"Plate {idx + 1}", use_column_width=True)

if __name__ == "__main__":
    main()