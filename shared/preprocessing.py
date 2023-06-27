import cv2
import numpy as np

def preprocess_image(image: np.ndarray) -> np.ndarray:
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Blur image
    blur = cv2.GaussianBlur(gray, (5,5), 0) 
    
    return blur

def read_image_and_preprocess(image_path: str) -> np.ndarray:
    #image = read_image_and_scale(image_path)
    image = read_image(image_path)
    
    preprocessed_image = preprocess_image(image)
    
    return preprocessed_image

def scale_image(image: np.ndarray) -> np.ndarray:
    # Desired maximum size
    max_size = 480

    # Calculate the ratio of the new size to the old size
    old_size = image.shape[:2]
    ratio = float(max_size) / max(old_size)

    # Calculate the new dimensions
    new_dim = (int(image.shape[1] * ratio), int(image.shape[0] * ratio))
    
    resized_image = cv2.resize(image, new_dim)
    
    delta_w = max_size - new_dim[1]
    delta_h = max_size - new_dim[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    color = [255, 255, 255]
    resized_image_with_border = cv2.copyMakeBorder(resized_image, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                                   value=color)
    
    #cv2.imshow('ROI', resized_image_with_border)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
        
    return resized_image_with_border

def read_image_and_scale(image_path: str) -> np.ndarray:
    image = read_image(image_path)
    return scale_image(image)


def read_image(image_path: str) -> np.ndarray:
    image = cv2.imread(image_path)
    return image.copy()
