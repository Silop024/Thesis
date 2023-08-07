import os
import configparser
import joblib

import cv2
from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator

from feature_extraction import extract_all_features, get_contours
from features import FeatureType
from processing import scale_data
import preprocessing
from shapes import Shape, ShapePrediction

def classify(image_path: str, clf: BaseEstimator, pca: PCA):
    image = preprocessing.read_image(image_path)
    
    preprocessed_image = preprocessing.preprocess_image(image)
    
    X, Y = extract_all_features(("ERROR", preprocessed_image), FeatureType.HOG)
    
    X, _ = scale_data(X, Y)
    
    X = pca.transform(X)
    
    predictions = clf.predict(X)
    
    contours = get_contours(preprocessed_image)
    
    centroids = []
    for contour in contours:
        moments = cv2.moments(contour)
        cx = int(moments['m10'] / moments['m00'])
        cy = int(moments['m01'] / moments['m00'])
        centroids.append((cx, cy))
        
    something = [] 
    for i, centroid in enumerate(centroids):
        something.append(
            ShapePrediction(
                index = i, 
                position = centroid, 
                shape = Shape(predictions[i]),
                probability = 1,
                id = 0, 
                image_path = image_path
            )
        )
        
    for prediction in something:
        cv2.putText(preprocessed_image, prediction.__repr__(), prediction.position, cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.4, color=(0, 0, 255), thickness=1)
            
    cv2.imshow('Predictions', preprocessed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # Read configs
    config = configparser.ConfigParser()
    config.read('config.ini') 
    model_dir = config['Paths']['model_dir']
    
    image_path = input("Enter image path: ")

    clf_path = os.path.join(model_dir, "clf.joblib")
    
    clf = joblib.load(os.path.join(model_dir, "clf.joblib"))
    pca = joblib.load(os.path.join(model_dir, "pca.joblib"))
    
    classify(image_path, clf, pca)