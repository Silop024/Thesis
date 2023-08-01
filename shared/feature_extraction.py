import cv2
import numpy as np
import mahotas.features
import skimage.feature

from typing import List


def get_hu_moments(roi: np.ndarray) -> np.ndarray:
    moments = cv2.moments(roi)
    
    hu_moments = cv2.HuMoments(moments)
    
    non_zero_mask = hu_moments != 0
    
    hu_moments[non_zero_mask] = -1 * np.sign(hu_moments[non_zero_mask]) * np.log10(np.abs(hu_moments[non_zero_mask]))
    
    return hu_moments.flatten()


def get_zernike_moments(contour: np.ndarray, image: np.ndarray) -> np.ndarray:
    _, radius = cv2.minEnclosingCircle(contour)
    
    roi = contour_to_roi(contour, image)
    
    zerinke_moments = mahotas.features.zernike_moments(roi, radius=radius)
    
    return zerinke_moments


def get_sift(roi: np.ndarray) -> np.ndarray:
    sift = cv2.SIFT_create()
    _, descriptors = sift.detectAndCompute(roi, None)
    
    return descriptors
    
    
def get_hog(roi: np.ndarray) -> np.ndarray:
    hog = skimage.feature.hog(roi, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2', feature_vector=True)
    
    return hog


def contour_to_roi(contour, image: np.ndarray) -> np.ndarray:
    # Reduce image to just the bounded area of the contour
    x, y, w, h = cv2.boundingRect(contour)
    roi = image[y:y + h, x:x + w]
    
    # Define maximum dimensions of the ROI
    max_h = 100
    max_w = 100
    
    # Create a blank image (padded_roi) of the maximum dimensions
    padded_roi = np.zeros((max_h, max_w), dtype=image.dtype)
    
    # Calculate the offset to place the ROI in the center of padded_roi
    offset_x = (max_w - w) // 2
    offset_y = (max_h - h) // 2
    
    # Place the ROI in the center of padded_roi image
    padded_roi[offset_y:offset_y + h, offset_x:offset_x + w] = roi
        
    return padded_roi


def get_contours(image: np.ndarray) -> List:
    # Get all contours in binary image
    contours, _ = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Remove contour that appears around the entire image for some unknown reason
    max_contour_area = 0.9 * image.shape[0] * image.shape[1]
    contours = [cnt for cnt in contours if cv2.contourArea(cnt) < max_contour_area]
    
    return contours


def get_regions_of_interest(image: np.ndarray) -> List[np.ndarray]:
    contours = get_contours(image)
    
    # Convert contours to regions of interest.
    rois = []
    for contour in contours:
        roi = contour_to_roi(contour, image)
        rois.append(roi)
        
    return rois
        





    
    