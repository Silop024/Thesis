# My own files
from feature_extraction import extract_all_features, get_contours
from features import FeatureType
import processing
import preprocessing

# Python standard libraries
import os
import configparser
from typing import List, Tuple

# Installed with pip
import cv2
import joblib
import matplotlib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator

matplotlib.rcParams['figure.figsize'] = (20.0, 10.0)
matplotlib.rcParams['image.cmap'] = 'gray'


def classify(image_path: str, clf: BaseEstimator, pca: PCA):
    image = preprocessing.read_image(image_path)
    
    preprocessed_image = preprocessing.preprocess_image(image)
    
    X, _ = extract_all_features(("ERROR", preprocessed_image), FeatureType.HOG)
    
    # Process data
    X = processing.scale_data(X)
    X = processing.use_pca(X, pca)
    
    # Classify, returns a list of labels
    #predictions = clf.predict(X)
    
    contours = get_contours(preprocessed_image)        
    
    backtorgb = cv2.cvtColor(preprocessed_image, cv2.COLOR_GRAY2RGB)
    
    contours_sorted = sort_shapes(contours[:-1])
    
    for i, c in enumerate(contours_sorted):
        x, y, w, h = c
        
        cv2.rectangle(backtorgb, (x, y), (x + w, y + h), (255,0,0), 2)
        cv2.putText(backtorgb, str(i), (x + int(w / 2), y + int(h / 2)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
    cv2.imshow("", backtorgb)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    


def sort_shapes(contours: List) -> List[Tuple]:
    bboxes = [cv2.boundingRect(i) for i in contours]
    
    c = np.array(bboxes)
    max_height = np.max(c[::, 3])
    # Sort the contours by y-value
    by_y = sorted(bboxes, key=lambda x: x[1])  # y values

    line_y = by_y[0][1]       # first y
    line = 1
    by_line = []

    # Assign a line number to each contour
    for x, y, w, h in by_y:
        if y > line_y + max_height:
            line_y = y
            line += 1
        
        by_line.append((line, x, y, w, h))

    # This will now sort automatically by line then by x
    contours_sorted = [(x, y, w, h) for line, x, y, w, h in sorted(by_line)]
    
    return contours_sorted
	


if __name__ == '__main__':
    # Read configs
    config = configparser.ConfigParser()
    config.read('config.ini') 
    model_dir = config['Paths']['model_dir']
    
    image_path = 'C:\\Users\\Silop\\Desktop\\Thesis\\test_images\\test_code-example-2.jpg'

    clf_path = os.path.join(model_dir, "clf.joblib")
    
    clf = joblib.load(os.path.join(model_dir, "clf.joblib"))
    pca = joblib.load(os.path.join(model_dir, "pca.joblib"))
    
    classify(image_path, clf, pca)