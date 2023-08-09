# Installed with pip
import cv2
import numpy as np
from cv2 import aruco


def preprocess_image(image: np.ndarray) -> np.ndarray:
    preprocessed_image = enhance_image(image)
    
    preprocessed_image = morph_image(preprocessed_image)
    
    preprocessed_image = scale_image(preprocessed_image)
    
    preprocessed_image = segment_image(preprocessed_image)
    
    return preprocessed_image


def scale_image(image: np.ndarray) -> np.ndarray:
    width_scale, height_scale = detect_image_scale(image)
    
    new_dim = (int(image.shape[0] / width_scale), int(image.shape[1] / height_scale))

    resized_image = cv2.resize(src=image, dsize=new_dim)
    
    return resized_image


def enhance_image(image: np.ndarray) -> np.ndarray:
    enhanced_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #enhanced_image = cv2.GaussianBlur(enhanced_image, ksize=(1, 1), sigmaX=1.0)
    #enhanced_image = cv2.addWeighted(enhanced_image, 2.0, enhanced_image, -1.0, 0)
    
    return enhanced_image


def morph_image(image: np.ndarray) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    morphed_image = cv2.morphologyEx(src=image, op=cv2.MORPH_OPEN, kernel=kernel)
    
    return morphed_image


def segment_image(image: np.ndarray) -> np.ndarray:
    _, segmented_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return segmented_image


def detect_image_scale(image: np.ndarray) -> tuple[int, int]:
    dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)
    params =  aruco.DetectorParameters()
    detector = aruco.ArucoDetector(dict, params)
    
    marker_corners, _, _ = detector.detectMarkers(image)
    
    if len(marker_corners) == 0:
        return 1, 1
    
    marker_width_px = np.linalg.norm(marker_corners[0][0][0] - marker_corners[0][0][1])
    marker_height_px = np.linalg.norm(marker_corners[0][0][0] - marker_corners[0][0][3])
    
    # Marker is the size of a 6x6 lego brick, thus each 1x1 lego brick, in pixels will be:
    width_scale = marker_width_px / 60
    height_scale = marker_height_px / 60
    
    marker_corners = np.array(marker_corners, dtype=np.int32)
    
    cv2.fillPoly(image, marker_corners, (255, 255, 255))

    return width_scale, height_scale


def read_image(image_path: str) -> np.ndarray:
    image = cv2.imread(image_path)
    
    return image.copy()
