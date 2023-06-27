import cv2
import numpy as np
from shared.preprocessing import read_image_and_preprocess
from sklearn.cluster import DBSCAN
    
     
def read_image_and_process(image_path: str, debug: bool = False) -> dict[str, np.ndarray | list]:
    """
    Reads, processes an image and computes Hu moments and centroids of contours.
    
    Args:
        image_path (str): Path to the image file.
        debug (bool, optional): If True, shows intermediate processing stages. Default is False.
        
    Returns:
        dict: A dictionary with two keys: 'humoments' and 'centroids' each containing a list of corresponding values.
    """
    processed_image = read_image_and_preprocess(image_path)

    # Use threshold for image segmentation.
    _, thresh = cv2.threshold(processed_image, 64, 255, cv2.THRESH_BINARY)

    # Get contours, but avoid the contour around the entire image and non-shapes by defining min and max size.
    min_contour_area = 20.0
    max_contour_area = 0.9 * processed_image.shape[0] * processed_image.shape[1]
    contours = get_contours(thresh, min_contour_area, max_contour_area)

    if debug:
        debug_contours(processed_image, contours)
        
    hu_moments_list: list[np.ndarray] = []
    centroids: list[tuple] = []
    for contour in contours:
        # Calculate the moments of the contour
        M = cv2.moments(contour)
        
        centroid = get_centroid(M)
        centroids.append(centroid)

        huMoments = get_hu_moments(M)
        hu_moments_list.append(huMoments)
    
    X = np.array([huMoments.flatten() for huMoments in hu_moments_list])
    #X = np.array(hu_moments_list)
        
    return {'humoments': X, 'centroids': centroids}

def get_contours(segments: np.ndarray, min_area: float, max_area: float) -> list:
    """Returns the contours found in a segmented image.

    Args:
        segments (np.ndarray): The segmented image, which can be generated by any image segmentation process (like cv2.threshold, cv2.Canny, etc.).
        min_area (float): The smallest area that a shape can have to be considered.
        max_area (float): The largest area that a shape can have to be considered.

    Returns:
        list: The contours of each segment in the image.
    """    
    contours, _ = cv2.findContours(segments, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = [cnt for cnt in contours if cv2.contourArea(cnt) < max_area and cv2.contourArea(cnt) > min_area]
    
    return contours


def get_centroid(moments) -> tuple:
    if moments["m00"] != 0: # avoid division by zero
        cX = int(moments["m10"] / moments["m00"])
        cY = int(moments["m01"] / moments["m00"])
        return (cX, cY)
    else:
        return (0, 0)


def get_hu_moments(moments) -> np.ndarray:
    # Calculate Hu Moments
    huMoments: np.ndarray = cv2.HuMoments(moments)

    # Convert Hu Moments to log scale, avoid 0 values to avoid value errors
    huMoments = huMoments + 1e-10
    huMoments = -1 * np.sign(huMoments) * np.log10(np.abs(huMoments))

    return huMoments


def cluster_analysis(image_path: str, clusterer: DBSCAN):
    processed_image = read_image_and_process(image_path)
    
    ids = clusterer.fit_predict(processed_image['humoments'])
    
    return ids
    

            
def debug_contours(image: np.ndarray, contours):
    image_copy = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    image_contours = cv2.drawContours(image_copy, contours, -1, (0,255,0), 1)
    
    cv2.imshow('Contours', image_contours)
    cv2.waitKey(0)
    cv2.destroyAllWindows()