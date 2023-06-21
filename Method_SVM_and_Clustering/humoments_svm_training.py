import os
import joblib
import configparser
import numpy as np
from sklearn import svm
from sklearn import preprocessing
from processing import read_image_and_process


config = configparser.ConfigParser()
config.read('config.ini')

# Directory where the trained model will be stored
model_dir = config.get('Paths', 'model_dir')

# Directory where the training images are stored
image_dir = config.get('Paths', 'training_dir')

# Get a list of all the image files in the directory
image_files = os.listdir(image_dir)

# Prepare empty lists for the feature vectors and labels
X = []  # feature vectors (Hu Moments)
Y = []  # labels (shape names)

# Process each image file
for image_file in image_files:
    hu_moments_list = read_image_and_process(f'{image_dir}{image_file}', debug=False)['humoments']
    shape_name = image_file.split('_')[0]
    
    # Flatten the Hu Moments arrays and add it to the list of feature vectors
    for hu_moments in hu_moments_list:
        # Flatten the Hu Moments arrays and add it to the list of feature vectors
        X.append(hu_moments.flatten())
        
        # Add name to the list of labels
        Y.append(shape_name)
        scaler = preprocessing.StandardScaler().fit(X)

X = np.array(X)
# Create a SVM classifier and train it on the training data
multi_clf = svm.SVC(C=10, kernel='poly', gamma=10) # poly and linear seem to be the most accurate
multi_clf.fit(X, Y)

one_clf = svm.OneClassSVM(kernel='poly', gamma=10)
one_clf.fit(X)

# Save model
joblib.dump(multi_clf, f'{model_dir}humoments_multi_svm_model.pkl')
joblib.dump(one_clf, f'{model_dir}humoments_one_svm_model.pkl')
