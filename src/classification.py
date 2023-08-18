# My own files
import parsing
import processing
import preprocessing
from features import FeatureType
import feature_extraction as extraction

# Python standard libraries
import os
import configparser

# Installed with pip
import bfi
import joblib
from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler, LabelEncoder


def classify(image_path: str, clf: BaseEstimator, pca: PCA, 
             scaler: StandardScaler, encoder: LabelEncoder, 
             debug=False):
    image = preprocessing.read_image(image_path)
    
    preprocessed_image = preprocessing.preprocess_image(image)
    
    X, _ = extraction.extract_all_features(("N/A", preprocessed_image), FeatureType.HOG)
    
    # Process data
    X = processing.scale_data(X, scaler)
    X = processing.use_pca(X, pca)
    
    # Classify, returns a list of labels
    predictions = clf.predict(X)
    
    if debug:
        extraction.show_features(X, predictions, pca)
    
    return predictions


if __name__ == '__main__':
    # Read configs
    config = configparser.ConfigParser()
    config.read('config.ini') 
    model_dir = config['Paths']['model_dir']
    models = config['Models']
    test_dir = config['Paths']['test_dir']
    
    image_path = os.path.join(test_dir, "HelloWorld.jpg")
    
    clf = joblib.load(os.path.join(model_dir, models['clf']))
    pca = joblib.load(os.path.join(model_dir, models['pca']))
    
    predictions = classify(image_path, clf, pca, True)
    
    hello_world = '++++++++[>++++[>++>+++>+++>+<<<<-]>+>+>->>+[<]<-]>>.>---.+++++++..+++.>>.<-.<.+++.------.--------.>>+.>++.'
    
    code = parsing.parse(predictions, hello_world)
    
    print(code)
    
    bfi.interpret(program=code, time_limit=5)
    