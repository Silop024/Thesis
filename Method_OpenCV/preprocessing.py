import cv2
import numpy as np

def preprocess_image(image: np.ndarray) -> np.ndarray:
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Blur image
    blur = cv2.GaussianBlur(gray, (5,5), 0) 
    
    return blur

def read_image_and_preprocess(image_path: str) -> np.ndarray:
    image = read_image_and_scale(image_path)
    
    preprocessed_image = preprocess_image(image)
    
    return preprocessed_image

def scale_image(image: np.ndarray) -> np.ndarray:
    # Desired maximum size
    max_size = 480

    # Calculate the ratio of the new size to the old size
    ratio = max_size / max(image.shape)

    # Calculate the new dimensions
    dim = (int(image.shape[1] * ratio), int(image.shape[0] * ratio))
    return cv2.resize(image, dim)

def read_image_and_scale(image_path: str) -> np.ndarray:
    image = cv2.imread(image_path)
    image_copy = image.copy()
    
    return scale_image(image_copy)