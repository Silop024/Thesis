import os
import joblib
import configparser
import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits import mplot3d

from sklearn import svm
from sklearn import preprocessing
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import PCA

# My libraries
from svm.processing import read_image_and_process
from shared.debugging import Debug

# For terminal output
from tqdm import tqdm


def train():
    Debug.print('Reading data from configs', level=1, color='red')

    config = configparser.ConfigParser()
    config.read('config.ini')

    # Directory where the trained model will be stored
    model_dir = config['Paths']['model_dir']

    # Directory where the training images are stored
    image_dir = config['Paths']['training_dir']


    # Get a list of all the image files in the directory
    image_files = os.listdir(image_dir)
    

    Debug.print('Reading data from configs... Done', level=1, color='green')
    Debug.print('Processing images...', level=1, color='red')

    # Prepare empty lists for the feature vectors and labels
    X = []  # feature vectors (Hu Moments)
    Y = []  # labels (shape names)
    
    if Debug.verbosity > 1:
        image_files = tqdm(image_files, colour='green')

    # Process each image file
    for image_file in image_files:
        hu_moments_array = read_image_and_process(os.path.join(image_dir, image_file), debug=Debug.verbosity > 2)['humoments']
        shape_name = image_file.split('_')[0]

        # Put shape labels in Y and hu moments in X
        for hu_moments in hu_moments_array:
            X.append(hu_moments)
            Y.append(shape_name)


    Debug.print('Processing images... Done', level=1, color='green')
    Debug.print('Standardizing features...', level=1, color='red')

    # Standardize features by removing the mean and scaling to unit variance
    scaler = preprocessing.StandardScaler()
    X = scaler.fit_transform(X)
    
    
    pca = PCA(n_components=3)
    pca.fit(X)
    X_pca = pca.transform(X)
    
    # Shows the variance ratio as the number of features changes.
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('# Components')
    plt.ylabel('Explained variance')
    plt.show()
    
    # 3d scatterplot
    # Color the data for the scatterplot
    colors = []
    for label in Y:
        match label:
            case '1x1':
                colors.append('#ff0000') # red
            case '1x2':
                colors.append('#00ff00') # lime green
            case '1x3':
                colors.append('#0000ff') # blue
            case '1x4':
                colors.append('#ffff00') # yellow
            case '2x2': 
                colors.append('#ff00ff') # magent
            case '2x3':
                colors.append('#00ffff') # cyan
            case '2x4':
                colors.append('#800080') # purple
    
    
    fig = plt.figure(figsize=(12, 8))
    ax = plt.axes(projection='3d')
    sctt = ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], s= 50, alpha=0.6, c=colors)
    plt.title('3D scatterplot: 83 percent of the variability captured', pad=15)
    ax.set_xlabel('First feature')
    ax.set_ylabel('Second feature')
    ax.set_zlabel('Third feature')
    plt.show()
    
    


    Debug.print('Standardizing features... Done', level=1, color='green')
    Debug.print('Training classifiers...', level=1, color='red')

    # Create a SVM classifier and train it on the training data
    base_estimator = svm.SVC(kernel='rbf', # poly and linear seem to be the most accurate
                             C=10, gamma='auto', 
                             verbose=Debug.verbosity > 3) 

    # Create the CalibratedClassifierCV using the base estimator
    calibrated_clf = CalibratedClassifierCV(base_estimator)

    calibrated_clf.fit(X_pca, Y)
    base_estimator.fit(X_pca, Y)
    

    Debug.print('Training classifiers... Done', level=1, color='green')
    Debug.print('Saving models...', level=1, color='red')
    
    # Save model
    joblib.dump(calibrated_clf, os.path.join(model_dir, config['Models']['calibrated_classifier']))
    joblib.dump(base_estimator, os.path.join(model_dir, config['Models']['multi_classifier']))
    

    Debug.print('Saving models... Done', level=1, color='green')
    Debug.print('Training finished', level=0, color='green')
    


if __name__ == "__main__":
    train()