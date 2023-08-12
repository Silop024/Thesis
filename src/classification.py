# My own files
from feature_extraction import extract_all_features, get_contours, show_features
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
import numpy as np
from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator


def classify(image_path: str, clf: BaseEstimator, pca: PCA):
    image = preprocessing.read_image(image_path)
    
    preprocessed_image = preprocessing.preprocess_image(image)
    
    X, Y = extract_all_features(("N/A", preprocessed_image), FeatureType.HOG)
    
    # Process data
    X = processing.scale_data(X)
    X = processing.use_pca(X, pca)
    
    # Classify, returns a list of labels
    predictions = clf.predict(X)
    
    #show_features(X, predictions, pca)
    
    config = configparser.ConfigParser()
    config.read('config.ini')
    shape_mapping_dict = config['Shapes']
    shape_mapping = {k: v for k, v in shape_mapping_dict.items()}
    
    symbol_list = [shape_mapping[p] for p in predictions]
    
    code = ''.join(symbol_list)
    
    hello_world = '++++++++[>++++[>++>+++>+++>+<<<<-]>+>+>->>+[<]<-]>>.>---.+++++++..+++.>>.<-.<.+++.------.--------.>>+.>++.'
    
    for i in range(min(len(code), len(hello_world))):
        if code[i] != hello_world[i]:
            print(f"Strings are not equal at index {i}: '{code[i]}' vs '{hello_world[i]}'")
            
    print(hello_world)
    print(code)
    
    #assert(code == hello_world)
    
    """contours = get_contours(preprocessed_image)        
    
    backtorgb = cv2.cvtColor(preprocessed_image, cv2.COLOR_GRAY2RGB)
    
    for i, c in enumerate(contours):
        cv2.drawContours(backtorgb, [c], -1, (0, 255, 0), 2)
        
        cv2.imshow("", backtorgb)
        cv2.waitKey(0)
        cv2.destroyAllWindows()"""


if __name__ == '__main__':
    # Read configs
    config = configparser.ConfigParser()
    config.read('config.ini') 
    model_dir = config['Paths']['model_dir']
    
    image_path = 'C:\\Users\\Silop\\Desktop\\Thesis\\test_images\\HelloWorld.jpg'

    clf_path = os.path.join(model_dir, "clf.joblib")
    
    clf = joblib.load(os.path.join(model_dir, "clf.joblib"))
    pca = joblib.load(os.path.join(model_dir, "pca.joblib"))
    
    classify(image_path, clf, pca)
    