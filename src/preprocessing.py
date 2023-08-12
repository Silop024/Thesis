# Installed with pip
import cv2
import numpy as np
from cv2 import aruco


def preprocess_image(image: np.ndarray) -> np.ndarray:
    preprocessed_image = grayscale_image(image)
    
    preprocessed_image = scale_image(preprocessed_image)
    
    """cv2.imshow("", preprocessed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()"""
    
    preprocessed_image = enhance_image(preprocessed_image)
    
    """cv2.imshow("", preprocessed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()"""
    
    preprocessed_image = morph_image(preprocessed_image)
    
    preprocessed_image = segment_image(preprocessed_image)
    
    return preprocessed_image


def grayscale_image(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def enhance_image(image: np.ndarray) -> np.ndarray:
    enhanced_image = cv2.GaussianBlur(image, ksize=(3, 3), sigmaX=0)
    
    kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
    enhanced_image = cv2.filter2D(enhanced_image, -1, kernel)
    
    return enhanced_image


def morph_image(image: np.ndarray) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    morphed_image = cv2.morphologyEx(src=image, op=cv2.MORPH_OPEN, kernel=kernel)
    
    return morphed_image


def segment_image(image: np.ndarray) -> np.ndarray:
    _, segmented_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    return segmented_image


def scale_image(image: np.ndarray) -> np.ndarray:
    width_scale, height_scale = detect_image_scale(image)
    
    new_dim = (int(image.shape[1] / width_scale), int(image.shape[0] / height_scale))

    resized_image = cv2.resize(src=image, dsize=new_dim, interpolation=cv2.INTER_CUBIC)
    #resized_image = cv2.resize(src=image, dsize=None, fx=width_scale, fy=height_scale)
    
    return resized_image


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
    width_scale =  marker_width_px / 60
    height_scale = marker_height_px / 60
    
    # Calculate the center of the marker
    marker_center = np.mean(marker_corners[0][0], axis=0, dtype=np.int32)

    # Define the size increase factor
    size_increase_factor = 2.2  # Adjust this factor as needed

    # Calculate the coordinates of the slightly bigger white square
    top_left = (marker_center[0] - int(marker_width_px * (size_increase_factor - 1) / 2),
                marker_center[1] - int(marker_height_px * (size_increase_factor - 1) / 2))
    bottom_right = (marker_center[0] + int(marker_width_px * (size_increase_factor - 1) / 2),
                    marker_center[1] + int(marker_height_px * (size_increase_factor - 1) / 2))

    # Ensure coordinates are within image bounds
    top_left = np.maximum(top_left, 0)
    bottom_right = np.minimum(bottom_right, (image.shape[1], image.shape[0]))

    # Create and fill the slightly bigger white square
    cv2.rectangle(image, tuple(top_left), tuple(bottom_right), (255, 255, 255), thickness=-1)

    return width_scale, height_scale


def read_image(image_path: str) -> np.ndarray:
    image = cv2.imread(image_path)
    
    return image.copy()
