import cv2
import os

# Ensure the file path is correct
mouth_cascade_path = r"D:\\proj4\\Drowsiness Detection\\haarcascade_mcs_mouth.xml"

# Load the cascade
mouth_cascade = cv2.CascadeClassifier(mouth_cascade_path)

# Check if the file is loaded correctly
if mouth_cascade.empty():
    print("Error: Mouth cascade classifier XML file not loaded properly!")
    exit()
