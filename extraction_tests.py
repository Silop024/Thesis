import shared.feature_extraction as extraction
import shared.preprocessing

import configparser
import os
import cv2
import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.manifold import TSNE

import sklearn.preprocessing
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt

from typing import Tuple


def test_hu_moments(images: Tuple[str, np.ndarray]):
    X = []  # features
    Y = []  # labels
    
    for shape_name, preprocessed_image in images: 
        #rois = extraction.get_regions_of_interest(preprocessed_image)
        rois = extraction.get_contours(preprocessed_image)
        
        for roi in rois:
            features = shared.feature_extraction.get_hu_moments(roi)
            
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
        contours = extraction.get_contours(preprocessed_image)
        
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
    descriptors = []
    for _, preprocessed_image in images:
        rois = extraction.get_regions_of_interest(preprocessed_image)
        
        for roi in rois:
            features = extraction.get_sift(roi)
            
            if features is not None:
                descriptors.extend(features)

    clusters = 100
    kmeans = KMeans(n_clusters=clusters)
    kmeans.fit(descriptors)
    codebook = kmeans.cluster_centers_

    X = []
    Y = []
    for shape_name, preprocessed_image in images:
        rois = extraction.get_regions_of_interest(preprocessed_image)
        
        for roi in rois:
            descriptors = extraction.get_sift(roi)
            
            if descriptors is None or descriptors.shape[0] <= 0:
                continue
            
            histogram = np.zeros(codebook.shape[0])
            
            distances = np.linalg.norm(descriptors[:, None] - codebook, axis=-1)
            
            nearest_clusters = np.argmin(distances, axis=1)
            
            for cluster_index in nearest_clusters:
                histogram[cluster_index] += 1
            
            histogram_sum = np.sum(histogram)
            
            if histogram_sum != 0:
                histogram /= histogram_sum
            histogram = np.array(histogram)
            
            X.append(histogram)
            Y.append(shape_name)

    X = np.array(X)
    Y = np.array(Y)
    
    print(X.shape)
    print(Y.shape)
    """
    unique_labels = np.unique(Y)
    for label in unique_labels:
        mean_histogram = X[Y == label].mean(axis=0)
        plt.bar(range(len(mean_histogram)), mean_histogram, alpha=0.5, label=f'Shape {label}')
    plt.title('Mean Histograms by Label')
    plt.xlabel('Bin')
    plt.ylabel('Count')
    plt.legend()
    plt.show()
    
    for label in unique_labels:
        distances = pdist(X[Y == label], metric='euclidean')
        plt.hist(distances, bins=50, alpha=0.5, label=f'Shape {label}')
    plt.title('Histograms of Histogram Distances by Label')
    plt.xlabel('Distance')
    plt.ylabel('Count')
    plt.legend()
    plt.show()
    
    tsne = TSNE(n_components=2)
    X_tsne = tsne.fit_transform(X)

    for label in unique_labels:
        plt.scatter(X_tsne[Y == label, 0], X_tsne[Y == label, 1], label=f'Shape {label}')
    plt.title('t-SNE Projection of Histograms by Label')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.legend()
    #plt.show()

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    for label in unique_labels:
        plt.scatter(X_pca[Y == label, 0], X_pca[Y == label, 1], label=f'Shape {label}')
    plt.title('PCA Projection of Histograms by Label')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.legend()
    plt.show()"""
    
    #tsne = TSNE(n_components=3)
    #X = tsne.fit_transform(X)
    pca = PCA(n_components=3)
    X = pca.fit_transform(X)
    
    show_features(X, Y)
    

def test_hog(images: Tuple[str, np.ndarray]):
    X = []  # features
    Y = []  # labels
    
    for shape_name, preprocessed_image in images:
        rois = extraction.get_regions_of_interest(preprocessed_image)
        
        for roi in rois:
            features = shared.feature_extraction.get_hog(roi)
            
            X.append(features)
            Y.append(shape_name)

    scaler = sklearn.preprocessing.StandardScaler()
    X = np.array(X)
    X = scaler.fit_transform(X)
    
    pca = PCA(n_components=20)
    X = pca.fit_transform(X)
        
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
    #test_hog(images)
    test_sift(images)


def show_features(X, Y):
    # Plot the scaled features
    num_features = X.shape[1]
    num_labels = len(np.unique(Y))
    num_rows = int(np.ceil(np.sqrt(num_features)))
    num_cols = int(np.ceil(num_features / num_rows))

    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(15, 12))
    axes_flat = axes.ravel()

    ax: np.ndarray
    for i in range(num_features):
        ax = axes_flat[i]
        for label in np.unique(Y):
            mask = Y == label
            ax.scatter(np.arange(X.shape[0])[mask], X[:, i][mask], label=f'Shape {label}')

        ax.set_title(f'Feature {i}')

    for i in range(num_features, num_rows * num_cols):
        axes_flat[i].axis('off')

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(labels))
    plt.tight_layout()
    plt.show()



if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read('config.ini')
    image_dir = config['Paths']['training_dir']
    image_paths = [os.path.join(image_dir, file) for file in os.listdir(image_dir)]
    
    test_all(image_paths)
    