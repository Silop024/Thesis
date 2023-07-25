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
    center, radius = cv2.minEnclosingCircle(roi)
    
    mask = np.ones_like(image) * 255
    cv2.drawContours(mask, [roi], -1, 0, -1)
    (x, y, w, h) = cv2.boundingRect(roi)
    roi = mask[y:y + h, x:x + w]
   
    #print(mask.shape)
    #print(image.shape)
    #print(roi.shape)
    
    #cv2.imshow('Mask Image', mask)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    
    #roi = roi.squeeze()
    
    zerinke_moments = mahotas.features.zernike_moments(mask, radius=radius, cm=center)
    
    return zerinke_moments


def get_sift(roi: np.ndarray, image: np.ndarray) -> np.ndarray:
    sift = cv2.SIFT_create()
    
    mask = np.ones_like(image) * 255
    cv2.drawContours(mask, [roi], -1, 0, -1)
    (x, y, w, h) = cv2.boundingRect(roi)
    roi = mask[y:y + h, x:x + w]
    
    keypoints = sift.detect(roi, None)
    
    return keypoints
    
    
def get_hog(roi: np.ndarray) -> np.ndarray:
    hog, _ = skimage.feature.hog(roi, pixels_per_cell=(8, 8), cells_per_block=(2, 2))
    
    return hog
