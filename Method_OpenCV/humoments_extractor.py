import cv2
from cv2 import Mat
import numpy as np
from preprocessor import preprocess
    
     
def getHuMoments(image_path: str, debug: bool):
    # Load image and make a copy to avoid modifying the image.
    image_copy = preprocess(image_path)
    
    # Convert the image to gray scale
    gray = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)

    # Blur the image
    blur = cv2.GaussianBlur(gray, (5,5), 0) 

    # Use threshold for image segmentation.
    _, thresh = cv2.threshold(blur, 127, 255, 0)

    # Get contours, but avoid the contour around the entire image and non-shapes.
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    min_contour_area = 50
    contours = [cnt for cnt in contours if cv2.contourArea(cnt) < 0.9*image_copy.shape[0]*image_copy.shape[1] and cv2.contourArea(cnt) > min_contour_area]

    if debug:
        debugContours(image=image_copy, contours=contours)
        
    result = []
    centroids = []
    for contour in contours:
        # Calculate the moments of the contour
        M = cv2.moments(contour)
        
        # For debugging
        if M["m00"] != 0: # avoid division by zero
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            centroids.append((cX, cY))
        else:
            centroids.append((0, 0))  # or handle this case differently

        # Calculate Hu Moments
        huMoments = cv2.HuMoments(M)
        
        # Avoid zero value
        huMoments = huMoments + 1e-10

        # Log transform to make the values more understandable
        huMoments = -1 * np.sign(huMoments) * np.log10(np.abs(huMoments))
        
        result.append(huMoments)
        
    return [result, centroids]
    
def debugThreshold(thresh):
    cv2.imshow('Threshold', thresh)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
            
def debugContours(image: Mat, contours):
    image_contours = cv2.drawContours(image.copy(), contours, -1, (0,255,0), 3)
    
    cv2.imshow('Contours', image_contours)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
