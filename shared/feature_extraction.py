import cv2
import numpy as np
import mahotas.features
import skimage.feature


def get_hu_moments(roi: np.ndarray) -> np.ndarray:
    moments = cv2.moments(roi)
    
    hu_moments = cv2.HuMoments(moments)
    
    non_zero_mask = hu_moments != 0
    
    hu_moments[non_zero_mask] = -1 * np.sign(hu_moments[non_zero_mask]) * np.log10(np.abs(hu_moments[non_zero_mask]))
    
    return hu_moments.flatten()


def get_zernike_moments(roi: np.ndarray, image: np.ndarray) -> np.ndarray:
    _, radius = cv2.minEnclosingCircle(roi)
    
    mask = np.zeros_like(image)
    cv2.drawContours(mask, [roi], -1, color=255, thickness=cv2.FILLED)
    (x, y, w, h) = cv2.boundingRect(roi)
    roi = mask[y:y + h, x:x + w]
    
    zerinke_moments = mahotas.features.zernike_moments(roi, radius=radius)
    
    return zerinke_moments


def get_sift(roi: np.ndarray, image: np.ndarray) -> np.ndarray:
    mask = np.zeros_like(image)
    cv2.drawContours(mask, [roi], -1, color=255, thickness=cv2.FILLED)
    (x, y, w, h) = cv2.boundingRect(roi)
    roi = mask[y:y + h, x:x + w]
    
    sift = cv2.SIFT_create()
    _, descriptors = sift.detectAndCompute(roi, None)
    
    return descriptors
    
    
def get_hog(roi: np.ndarray, image: np.ndarray) -> np.ndarray:
    mask = np.zeros_like(image)
    cv2.drawContours(mask, [roi], -1, color=255, thickness=cv2.FILLED)
    (x, y, w, h) = cv2.boundingRect(roi)
    roi = mask[y:y + h, x:x + w]
    
    hog = skimage.feature.hog(roi, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2', feature_vector=True)
    
    return hog

