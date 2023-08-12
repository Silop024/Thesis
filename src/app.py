# My own files
import parsing
import classification

# Python standard libraries
import os
import configparser

# Installed with pip
import joblib
import bfi


if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read('config.ini') 
    model_dir = config['Paths']['model_dir']
    test_dir = config['Paths']['test_dir']
    
    clf = joblib.load(os.path.join(model_dir, "clf.joblib"))
    pca = joblib.load(os.path.join(model_dir, "pca.joblib"))
    
    hello_world = os.path.join(test_dir, 'HelloWorld.jpg')
    
    # Classify shapes
    predictions = classification.classify(image_path=hello_world, clf=clf, pca=pca, debug=False)
    
    code = parsing.parse(predictions)
    
    print(code)
    
    # Convert to code
    bfi.interpret(code)