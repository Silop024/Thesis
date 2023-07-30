import shared.feature_extraction
import shared.preprocessing

import configparser
import os
import cv2
import numpy as np

import sklearn.preprocessing
from sklearn.decomposition import PCA
from skimage import measure

import matplotlib.pyplot as plt

from typing import Callable, Tuple


def test_hu_moments(images: Tuple[str, np.ndarray]):
    X = []  # features
    Y = []  # labels
    
    for shape_name, preprocessed_image in images: 
        contours, _ = cv2.findContours(preprocessed_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        max_contour_area = 0.9 * preprocessed_image.shape[0] * preprocessed_image.shape[1]
        contours = [cnt for cnt in contours if cv2.contourArea(cnt) < max_contour_area]
        
        for contour in contours:
            features = shared.feature_extraction.get_hu_moments(contour)
            
            X.append(features)
            Y.append(shape_name)
            
    scaler = sklearn.preprocessing.StandardScaler()
    X = np.array(X)
    X = scaler.fit_transform(X)
    
    #pca = PCA(3)
    #X = pca.fit_transform(X)
        
    encoder = sklearn.preprocessing.LabelEncoder()
    Y_le = encoder.fit_transform(Y)
    Y = encoder.inverse_transform(Y_le)
    
    show_features(X, Y)
            

def test_zernike_moments(images: Tuple[str, np.ndarray]):
    X = []  # features
    Y = []  # labels
    
    for shape_name, preprocessed_image in images:
        contours, _ = cv2.findContours(preprocessed_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        max_contour_area = 0.9 * preprocessed_image.shape[0] * preprocessed_image.shape[1]
        contours = [cnt for cnt in contours if cv2.contourArea(cnt) < max_contour_area]
        
        for contour in contours:
            features = shared.feature_extraction.get_zernike_moments(contour, preprocessed_image)
            
            X.append(features)
            Y.append(shape_name)
            
    scaler = sklearn.preprocessing.StandardScaler()
    X = np.array(X)
    X = scaler.fit_transform(X)
    
    #pca = PCA(7)
    #X = pca.fit_transform(X)
        
    encoder = sklearn.preprocessing.LabelEncoder()
    Y_le = encoder.fit_transform(Y)
    Y = encoder.inverse_transform(Y_le)
    
    show_features(X, Y)
            
# TODO: Fix SIFT with Bag-Of-Words or K-Means clustering or other things.
def test_sift(images: Tuple[str, np.ndarray]):
    X = []  # features
    Y = []  # labels
    
    for shape_name, preprocessed_image in images:
        contours, _ = cv2.findContours(preprocessed_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        max_contour_area = 0.9 * preprocessed_image.shape[0] * preprocessed_image.shape[1]
        contours = [cnt for cnt in contours if cv2.contourArea(cnt) < max_contour_area]
        
        for contour in contours:
            features = shared.feature_extraction.get_sift(contour, preprocessed_image)
            
            X.append(features)
            Y.append(shape_name)

    scaler = sklearn.preprocessing.StandardScaler()
    X = np.array(X)
    X = scaler.fit_transform(X)
    
    #pca = PCA(7)
    #X = pca.fit_transform(X)
        
    encoder = sklearn.preprocessing.LabelEncoder()
    Y_le = encoder.fit_transform(Y)
    Y = encoder.inverse_transform(Y_le)
    
    show_features(X, Y)
    

def test_hog(images: Tuple[str, np.ndarray]):
    X = []  # features
    Y = []  # labels
    
    for shape_name, preprocessed_image in images:
        contours, _ = cv2.findContours(preprocessed_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        max_contour_area = 0.9 * preprocessed_image.shape[0] * preprocessed_image.shape[1]
        contours = [cnt for cnt in contours if cv2.contourArea(cnt) < max_contour_area]
        
        for contour in contours:
            features = shared.feature_extraction.get_hog(contour, preprocessed_image)
            
            if len(features) > 0:
                X.append(features)
                Y.append(shape_name)
                
    for elem in X:
        print(elem.shape)

    scaler = sklearn.preprocessing.StandardScaler()
    X = np.array(X)
    X = scaler.fit_transform(X)
        
    encoder = sklearn.preprocessing.LabelEncoder()
    Y_le = encoder.fit_transform(Y)
    Y = encoder.inverse_transform(Y_le)
    
    show_features(X, Y)

def test_all(image_paths):
    images = []
    for image_path in image_paths:
        image = shared.preprocessing.read_image(image_path)
        preprocessed_image = shared.preprocessing.preprocess_image(image)
        
        file_name = os.path.basename(image_path)
        shape_name = file_name.split('_')[0]
        
        images.append((shape_name, preprocessed_image))
    
    
    #test_hu_moments(images)
    #test_zernike_moments(images)
    test_hog(images)


def show_features(X, Y):
    # Plot the scaled features
    num_features = X.shape[1]
    num_labels = len(np.unique(Y))
    num_rows = int(np.ceil(np.sqrt(num_features)))
    num_cols = int(np.ceil(num_features / num_rows))

    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(15, 12))
    axes_flat = axes.ravel()

    for i in range(num_features):
        ax = axes_flat[i]
        for label in np.unique(Y):
            mask = Y == label
            ax.scatter(np.arange(X.shape[0])[mask], X[:, i][mask], label=f'Shape {label}')

        ax.set_title(f'Feature {i}')
        #ax.legend()

    for i in range(num_features, num_rows * num_cols):
        axes_flat[i].axis('off')

    plt.tight_layout()
    plt.show()



if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read('config.ini')
    image_dir = config['Paths']['training_dir']
    image_paths = [os.path.join(image_dir, file) for file in os.listdir(image_dir)]
    
    test_all(image_paths)
    