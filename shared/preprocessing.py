import cv2
import numpy as np
import skimage.filters

def scale_image(image: np.ndarray) -> np.ndarray:
    width_scale, height_scale = detect_image_scale(image)
    
    new_dim = (image.shape[0] / width_scale, image.shape[1] / height_scale)

    resized_image = cv2.resize(src=image, dsize=new_dim)
        
    return resized_image


def enhance_image(image: np.ndarray) -> np.ndarray:
    enhanced_image = skimage.filters.unsharp_mask(image)
    
    return enhanced_image


def morph_image(image: np.ndarray) -> np.ndarray:
    morphed_image = cv2.morphologyEx(src=image, op=cv2.MORPH_CLOSE)
    
    return morphed_image


def segment_image(image: np.ndarray) -> np.ndarray:
    _, segmented_image = cv2.threshold(image, 0, 255)
    
    return segmented_image


aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
def detect_image_scale(image: np.ndarray) -> tuple[int, int]:
    marker_params =  cv2.aruco.DetectorParameters_create()
    marker_corners, _, _ = cv2.aruco.detectMarkers(image, aruco_dict, parameters=marker_params)
    
    if len(marker_corners) == 0:
        exit(1)
    
    marker_width_px = np.linalg.norm(marker_corners[0][0][0] - marker_corners[0][0][1])
    marker_height_px = np.linalg.norm(marker_corners[0][0][0] - marker_corners[0][0][3])
    
    # Marker is the size of a 6x6 lego brick, thus each 1x1 lego brick, in pixels will be:
    width_scale = marker_width_px / 6
    height_scale = marker_height_px / 6

    return width_scale, height_scale


def read_image(image_path: str) -> np.ndarray:
    image = cv2.imread(image_path)
    return image.copy()
