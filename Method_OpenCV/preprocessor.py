import cv2
from cv2 import Mat

def preprocess(image_path: str) -> Mat:
    image = cv2.imread(image_path)
    image_copy = image.copy()
    
    # Scale the image
    scale_percent = 60 # percent of original size
    width = int(image_copy.shape[1] * scale_percent / 100)
    height = int(image_copy.shape[0] * scale_percent / 100)
    dim = (width, height)
    image_copy = cv2.resize(image_copy, dim) 
    
    return image_copy
    