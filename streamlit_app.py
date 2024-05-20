import cv2
import numpy as np
import pytesseract
import streamlit as st
import pandas as pd
from ultralytics import YOLO
import re

# Set Tesseract command to the correct path
# Set Tesseract command to default path in Google Colab
pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'
# pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract'

# Load the custom-trained YOLOv8 model
model = YOLO('best.pt')

def preprocess_license_plate(image):
    # Convert to grayscale
    gray_license_plate = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Resize the image to a fixed height while maintaining aspect ratio
    height = 100
    aspect_ratio = image.shape[1] / image.shape[0]
    width = int(aspect_ratio * height)
    gray_license_plate = cv2.resize(gray_license_plate, (width, height))
    
    # Apply bilateral filter to reduce noise while keeping edges sharp
    gray_license_plate = cv2.bilateralFilter(gray_license_plate, 11, 17, 17)
    
    # Apply adaptive thresholding to preprocess the license plate image
    _, license_plate_thresh = cv2.threshold(gray_license_plate, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    
    # Perform morphological operations to enhance the characters
    kernel = np.ones((3, 3), np.uint8)
    license_plate_thresh = cv2.morphologyEx(license_plate_thresh, cv2.MORPH_CLOSE, kernel)
    
    return license_plate_thresh

def process_frame(image):
    # Perform inference with YOLOv8
    results = model(image)
    
    recognized_text = ""
    boxes = []

    # Extract the bounding boxes, confidence scores, and class IDs
    for result in results:
        boxes.extend(result.boxes.xyxy.cpu().numpy())
    
    for box in boxes:
        if len(box) == 6:
            x1, y1, x2, y2, conf, class_id = box
        else:
            x1, y1, x2, y2 = box[:4]
            conf, class_id = None, None  # Set defaults if values are missing
        
        # Convert coordinates to integers
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        # Calculate the aspect ratio of the bounding box
        aspect_ratio = (x2 - x1) / (y2 - y1)
        
        # Check if aspect ratio is within a reasonable range for license plates
        if 2.5 < aspect_ratio < 8:
            # Crop the detected license plate region
            license_plate = image[y1:y2, x1:x2]
            
            if license_plate is not None:
                # Preprocess the license plate image
                license_plate_thresh = preprocess_license_plate(license_plate)
                
                # Use pytesseract for OCR to recognize text on the license plate
                custom_config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'  # OCR Engine Mode and Page Segmentation Mode with whitelist
                plate_text = pytesseract.image_to_string(license_plate_thresh, config=custom_config)
                
                # Filter out non-alphanumeric characters except common license plate characters
                plate_text = re.sub(r'[^A-Za-z0-9]', '', plate_text)
                plate_text = re.sub(r'[^A-Za-z0-9\-~`./|?()\[\]{}]:;','', plate_text)
                
                # Load the CSV file and check the recognized text
                df = pd.read_csv('recognized_text.csv')
                allowed_cars = df['Recognized Text'].tolist()
                
                if plate_text in allowed_cars:
                    status = "Vehicle from inside (allow vehicle)"
                else:
                    status = "Vehicle from outside (No entry)"
                
                st.write(status)
                
                # Draw bounding box and text on the image
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.putText(image, plate_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                
                print("License Plate:", plate_text)
                st.write(plate_text)
                
                recognized_text = plate_text

    return image, recognized_text

def save_to_csv(text):
    # Check if the CSV file exists, if not, create it with headers
    try:
        df = pd.read_csv('recognized_text.csv')
    except FileNotFoundError:
        df = pd.DataFrame(columns=['Recognized Text'])
    
    # Create a new DataFrame with the new row
    new_row = pd.DataFrame({'Recognized Text': [text]})
    
    # Concatenate the new DataFrame with the existing DataFrame
    df = pd.concat([df, new_row], ignore_index=True)
    
    # Save the DataFrame to CSV
    df.to_csv('recognized_text.csv', index=False)
    st.success("Text saved to recognized_text.csv")

def main():
    with open("style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    
    st.markdown(
        """
        <style>
        .main-header {font-size: 24px; color: #333;}
        .sidebar .sidebar-content {padding: 20px;}
        </style>
        """,
        unsafe_allow_html=True
    )
    
    # Apply the custom CSS
    cols1, cols2 = st.columns([1, 5])

    with cols1:
        st.image('./logo.jpeg')

    with cols2:
        st.write("# Rajkiya Engineering College Kannauj ")
    image_path = './Image.jpg'    
    st.sidebar.image(image_path, use_column_width=True)
    st.sidebar.write("## College Gate Security System :car:")
    st.sidebar.write("\n**Guided By:**  \nAshwini Kumar Upadhyaya  \nAsst. Professor, Rec Kannauj")
    st.sidebar.write("\n\n\n**Project By:**  \nKartik Rajput(33)  \nAviral Varshney (26)  \nAbhinav Rai(04)  \nShivam Singh(55)  \nAkash Verma(14)")

    st.header("College Gate Security System :car:")
    picture = st.camera_input("Click Image")

    if picture is not None:
        # Convert the uploaded image data to a numpy array
        image_array = np.frombuffer(picture.getvalue(), np.uint8)
        frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        
        # Process the frame
        p_image, recognized_text = process_frame(frame)
        
        # Display the processed frame in the Streamlit app
        st.image(p_image, channels="BGR", use_column_width=True)
        
        if st.button("Save Text to CSV"):
            save_to_csv(recognized_text)
        
        csv_file_path = "recognized_text.csv"
        
        # Read the CSV file
        df = pd.read_csv(csv_file_path)
        
        # Display the contents of the CSV file
        st.write(df)
    else:
        st.warning("Please upload an image.")

if __name__ == "__main__":
    main()



#streamlit run try_3.py --server.enableXsrfProtection false
# streamlit run streamlit_app.py --server.enableXsrfProtection false 
