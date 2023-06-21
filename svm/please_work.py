import os
import cv2
import joblib
import configparser
import numpy as np
from collections import Counter

from sklearn import svm, datasets
from sklearn import preprocessing
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder

from shared.preprocessing import read_image_and_preprocess



print('-------- Reading config file: Started --------')

config = configparser.ConfigParser()
config.read('config.ini')

# Directory where the trained model will be stored
model_dir = config['Paths']['model_dir']

# Directory where the training images are stored
image_dir = config['Paths']['training_dir']

print('-------- Reading config file: Done --------\n')

print('-------- Processing images: Started --------')

# Get a list of all the image files in the directory
image_files = os.listdir(image_dir)

# Prepare empty lists for the feature vectors and labels
X = []  # feature vectors (Hu Moments)
Y = []  # labels (shape names)

# Process each image file
for image_file in image_files:
    print(f'---------- {image_file} ----------')
    shape_name = image_file.split('_')[0]
    
    processed_image = read_image_and_preprocess(os.path.join(image_dir, image_file))
    
    _, thresh = cv2.threshold(processed_image, 64, 255, cv2.THRESH_BINARY)
    
    min_contour_area = 20.0
    max_contour_area = 0.9 * processed_image.shape[0] * processed_image.shape[1]
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = [cnt for cnt in contours if cv2.contourArea(cnt) < max_contour_area and cv2.contourArea(cnt) > min_contour_area]
    
    for contour in contours:
        # Calculate the moments of the contour
        M = cv2.moments(contour)
    
        hu_moments: np.ndarray = cv2.HuMoments(M)

        # Convert Hu Moments to log scale, avoid 0 values to avoid value errors
        hu_moments = hu_moments + 1e-10
        hu_moments = -1 * np.sign(hu_moments) * np.log10(np.abs(hu_moments))
        hu_moments = hu_moments.flatten()
        
        # Append to training data
        X.append(hu_moments)
        Y.append(shape_name)
        

print('-------- Processing images: Done --------\n')

# Standardize features by removing the mean and scaling to unit variance
scaler = preprocessing.StandardScaler()
X = np.array(X)
X = scaler.fit_transform(X)

# Create a SVM classifier and train it on the training data
parameters = {'estimator__C':[1, 10, 100], 'estimator__gamma':[0.1, 0.01]}

base_estimator = svm.SVC(kernel='poly') # poly and linear seem to be the most accurate
calibrated_clf = CalibratedClassifierCV(estimator=base_estimator)
grid_clf = GridSearchCV(estimator=calibrated_clf, param_grid=parameters, cv=5)

print('-------- Training classifier: Started --------')

le = LabelEncoder()
Y_le = le.fit_transform(Y)
original_labels = le.inverse_transform(Y_le)
print(Y_le)
# Count the occurrences of each class in the dataset
class_counts = Counter(original_labels)

# Print the classes that have less than 5 examples
small_classes = {class_name: count for class_name, count in class_counts.items() if count < 5}
print("Classes with less than 5 examples:", small_classes)

# Determine the maximum number of folds for cross-validation
max_folds = min(class_counts.values())
cv_folds = min(max_folds, 5)  # Ensure that the number of folds does not exceed 5
print(f'cv_folds: {cv_folds}, class_counts: {class_counts}')

# Create the CalibratedClassifierCV using the base estimator
#calibrated_clf = CalibratedClassifierCV(base_estimator, cv=10)

#calibrated_clf.fit(X, original_labels)
grid_clf.fit(X, original_labels)

print("Best parameters set found on development set:")
print(grid_clf.best_params_)


print('-------- Training classifier: Done --------\n')

# Save model
joblib.dump(grid_clf, os.path.join(model_dir, config['Models']['calibrated_classifier']))
