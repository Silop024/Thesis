from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import joblib
from humoments_extractor import getHuMoments
import os

# Directory where the trained model will be stored
model_dir = 'C:\\Users\\Jack\\Desktop\\Thesis\\Trained_Models\\'

# Directory where the training images are stored
image_dir = 'C:\\Users\\Jack\\Desktop\\Thesis\\Training_Images\\'

# Get a list of all the image files in the directory
image_files = os.listdir(image_dir)

# Prepare empty lists for the feature vectors and labels
X = []  # feature vectors (Hu Moments)
Y = []  # labels (shape names)

# Process each image file
for image_file in image_files:
    huMoments = getHuMoments(f'{image_dir}{image_file}', debug=False)[0]
    shape_name = image_file.split('_')[0]
    
    # Flatten the Hu Moments arrays and add it to the list of feature vectors
    for huMoment in huMoments:
        # Flatten the Hu Moments arrays and add it to the list of feature vectors
        X.append(huMoment.flatten())
        
        # Add name to the list of labels
        Y.append(shape_name)
        scaler = preprocessing.StandardScaler().fit(X)
    
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Create a SVM classifier and train it on the training data
clf = svm.SVC(C=10, kernel='poly', gamma=10) # poly and linear seem to be the most accurate
clf.fit(X_train, y_train)

# Save model
joblib.dump(clf, f'{model_dir}humoments_svm_model.pkl')
