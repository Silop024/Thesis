import cv2
import numpy as np
import mahotas.features
import skimage.feature

from typing import List, Tuple

from features import FeatureType


def get_hu_moments(roi: np.ndarray) -> np.ndarray:
    moments = cv2.moments(roi)
    
    hu_moments = cv2.HuMoments(moments)
    
    non_zero_mask = hu_moments != 0
    
    hu_moments[non_zero_mask] = -1 * np.sign(hu_moments[non_zero_mask]) * np.log10(np.abs(hu_moments[non_zero_mask]))
    
    return hu_moments.flatten()


def get_zernike_moments(roi: np.ndarray, radius: int) -> np.ndarray:
    zerinke_moments = mahotas.features.zernike_moments(roi, radius=radius)
    
    return zerinke_moments


def get_sift(roi: np.ndarray) -> np.ndarray:
    sift = cv2.SIFT_create()
    _, descriptors = sift.detectAndCompute(roi, None)
    
    return descriptors
    
    
def get_hog(roi: np.ndarray) -> np.ndarray:
    hog = skimage.feature.hog(roi, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2', feature_vector=True)
    
    return hog


def contour_to_roi(contour: np.array) -> np.ndarray:
     # Get bounding rectangle dimensions
    x, y, w, h = cv2.boundingRect(contour)
    
    # Define maximum dimensions of the ROI
    height = 100
    width = 100
    
    # Create a blank image (padded_roi) of the maximum dimensions
    padded_roi = np.zeros((height, width), dtype=np.uint8)
    
    # Calculate the offset to place the contour in the center of padded_roi
    offset_x = (width - w) // 2
    offset_y = (height - h) // 2
    
    # Create a new contour with updated coordinates
    new_contour = contour.copy()
    new_contour[:, 0, 0] = contour[:, 0, 0] - x + offset_x
    new_contour[:, 0, 1] = contour[:, 0, 1] - y + offset_y
    
    # Draw the contour on padded_roi
    cv2.drawContours(padded_roi, [new_contour], -1, 255, thickness=cv2.FILLED)
        
    return padded_roi


def get_contours(image: np.ndarray) -> List[np.array]:
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
        roi = contour_to_roi(contour)
        rois.append(roi)
        
    return rois
        
        
def create_codebook(images: List[Tuple[str, np.ndarray]], size=100) -> Tuple[np.ndarray, np.ndarray]:
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 0.01)
    flags = cv2.KMEANS_RANDOM_CENTERS
    
    descriptors = []
    for _, image in images:
        rois = get_regions_of_interest(image)
        
        for roi in rois:
            des = get_sift(roi)
        
            if des is not None:
                descriptors.extend(des)
        
    
    X = np.vstack(descriptors)
    
    _, labels, _ = cv2.kmeans(X, size, None, criteria, 10, flags)
    
    codebook = np.zeros((size, X.shape[1]))
    
    for i in range(size):
        codebook[i] = np.mean(X[labels.ravel() == i], axis=0)
        
    return codebook, descriptors

        
def extract_all_sift(image_tuple: Tuple[str, np.ndarray], codebook, descriptors) -> Tuple[np.array, np.array]:
    label, image = image_tuple
    
    X = []  # features
    Y = []  # labels
    
    for des in descriptors:
        histogram = np.zeros(codebook.shape[0])
        
        dists = np.linalg.norm(codebook - des, axis=1)
        
        min_idx = np.argmin(dists)
        
        histogram[min_idx] += 1
        
        X.append(histogram)
        Y.append(label)
    
    return X, Y
    
    
    
        

def extract_all_features(image_tuple: Tuple[str, np.ndarray], feature_type: FeatureType) -> Tuple[np.array, np.array]:
    if feature_type is FeatureType.SIFT:
        return extract_all_sift(image_tuple)
    
    label, image = image_tuple
   
    X = []  # features
    Y = []  # labels
    
    contours = get_contours(image)
    
    for contour in contours:
        roi = contour_to_roi(contour)
        
        match feature_type:
            case FeatureType.HuMoments:
                features = get_hu_moments(roi)
            case FeatureType.ZernikeMoments:
                _, radius = cv2.minEnclosingCircle(contour)
                features = get_zernike_moments(roi, radius)
            case FeatureType.HOG:
                features = get_hog(roi)
        
        X.append(features)
        Y.append(label)
    
    return X, Y



    
    