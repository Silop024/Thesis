# My own files
from feature_extraction import extract_all_features, get_contours, show_features
from features import FeatureType
import processing
import preprocessing
import parsing

# Python standard libraries
import os
import configparser

# Installed with pip
import cv2
import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator


def classify(image_path: str, clf: BaseEstimator, pca: PCA, debug=False):
    image = preprocessing.read_image(image_path)
    
    preprocessed_image = preprocessing.preprocess_image(image)
    
    X, Y = extract_all_features(("N/A", preprocessed_image), FeatureType.HOG)
    
    # Process data
    X = processing.scale_data(X)
    X = processing.use_pca(X, pca)
    
    # Classify, returns a list of labels
    predictions = clf.predict(X)
    
    if debug:
        show_features(X, predictions, pca)
    
    return predictions
    


if __name__ == '__main__':
    # Read configs
    config = configparser.ConfigParser()
    config.read('config.ini') 
    model_dir = config['Paths']['model_dir']
    test_dir = config['Paths']['test_dir']
    
    image_path = os.path.join(test_dir, "HelloWorld.jpg")

    clf_path = os.path.join(model_dir, "clf.joblib")
    
    clf = joblib.load(os.path.join(model_dir, "clf.joblib"))
    pca = joblib.load(os.path.join(model_dir, "pca.joblib"))
    
    predictions = classify(image_path, clf, pca, False)
    
    hello_world = '++++++++[>++++[>++>+++>+++>+<<<<-]>+>+>->>+[<]<-]>>.>---.+++++++..+++.>>.<-.<.+++.------.--------.>>+.>++.'
    
    code = parsing.parse(predictions, hello_world)
    
    print(code)
    
    parsing.parse(predictions, )
    