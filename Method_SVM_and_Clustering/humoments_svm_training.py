import os
import joblib
import configparser
from sklearn import svm
from sklearn import preprocessing
from sklearn.calibration import CalibratedClassifierCV
from processing import read_image_and_process
import shared.debugging as debugging

print('-------- Reading config file: Started --------')

loading = debugging.LoadingAnimation(message='Reading data from configs')
loading.start()

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

loading.stop()
loading.set_message('Processing images')
loading.start()

# Prepare empty lists for the feature vectors and labels
X = []  # feature vectors (Hu Moments)
Y = []  # labels (shape names)

# Process each image file
for image_file in image_files:
    hu_moments_array = read_image_and_process(os.path.join(image_dir, image_file), debug=False)['humoments']
    shape_name = image_file.split('_')[0]
    
    # Put shape labels in Y and hu moments in X
    for hu_moments in hu_moments_array:
        X.append(hu_moments)
        Y.append(shape_name)


print('-------- Processing images: Done --------\n')

# Standardize features by removing the mean and scaling to unit variance
scaler = preprocessing.StandardScaler()
X = scaler.fit_transform(X)


loading.stop()
loading.set_message('Training classifiers')
loading.start()

X = np.array(X)

# Create a SVM classifier and train it on the training data
base_estimator = svm.SVC(C=10, kernel='poly', gamma=10) # poly and linear seem to be the most accurate

print('-------- Training classifier: Started --------')

# Create the CalibratedClassifierCV using the base estimator
calibrated_clf = CalibratedClassifierCV(base_estimator, cv=5)

calibrated_clf.fit(X, Y)

# Train the classifier on the training data
base_estimator.fit(X, Y)

print('-------- Training classifier: Done --------\n')

# Save model
joblib.dump(calibrated_clf, os.path.join(model_dir, config['Models']['calibrated_classifier']))
joblib.dump(base_estimator, os.path.join(model_dir, config['Models']['multi_classifier']))

loading.stop()
