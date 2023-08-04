import shared.feature_extraction as extraction
import shared.preprocessing
from shared.features import FeatureType
import shared.processing as processing

import configparser
import os
import numpy as np

from sklearn.decomposition import PCA

import plotly.express as px

from typing import Tuple


def test_hu_moments(images: Tuple[str, np.ndarray]):
    X = []  # features
    Y = []  # labels
    
    for image_tuple in images: 
        x, y = extraction.extract_all_features(image_tuple, FeatureType.HuMoments)
        
        X.extend(x)
        Y.extend(y)
         
    X, Y = processing.scale_data(X, Y)
    
    n_components = 7
    pca = PCA(n_components=n_components)
    X = pca.fit_transform(X)
    
    show_features(X, Y, pca, n_components)
            

def test_zernike_moments(images: Tuple[str, np.ndarray]):
    X = []  # features
    Y = []  # labels
    
    for image_tuple in images:
        x, y = extraction.extract_all_features(image_tuple, FeatureType.ZernikeMoments)
        
        X.extend(x)
        Y.extend(y)
    
    X, Y = processing.scale_data(X, Y)
    
    n_components = 7
    pca = PCA(n_components=n_components)
    X = pca.fit_transform(X)
    
    show_features(X, Y, pca, n_components)
            
# TODO: Fix SIFT with Bag-Of-Words or K-Means clustering or other things.
def test_sift(images: list[Tuple[str, np.ndarray]]):
    """descriptors = []
    for _, preprocessed_image in images:
        rois = extraction.get_regions_of_interest(preprocessed_image)
        
        for roi in rois:
            features = extraction.get_sift(roi)
            
            if features is not None:
                descriptors.extend(features)

    clusters = 10
    kmeans = KMeans(n_clusters=clusters, n_init='auto')
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
            Y.append(shape_name)"""
            
    codebook, descriptors = extraction.create_codebook(images)
            
    X = []
    Y = []
    for image_tuple in images:
        x, y = extraction.extract_all_sift(image_tuple, codebook, descriptors)
        
        X.extend(x)
        Y.extend(y)

    X, Y = processing.scale_data(X, Y)
    
    n_components = 7
    pca = PCA(n_components=n_components)
    X = pca.fit_transform(X)
    
    show_features(X, Y, pca, n_components)
    

def test_hog(images: Tuple[str, np.ndarray]):
    X = []  # features
    Y = []  # labels
    
    for image_tuple in images:
        x, y = extraction.extract_all_features(image_tuple, FeatureType.HOG)
        
        X.extend(x)
        Y.extend(y)
            
    X, Y = processing.scale_data(X, Y)
    
    n_components = 7
    pca = PCA(n_components=n_components)
    X = pca.fit_transform(X)
    
    show_features(X, Y, pca, n_components)

def test_all(image_paths):
    images = []
    for image_path in image_paths:
        image = shared.preprocessing.read_image(image_path)
        preprocessed_image = shared.preprocessing.preprocess_image(image)
        
        file_name = os.path.basename(image_path)
        shape_name = file_name.split('_')[0]
        
        images.append((shape_name, preprocessed_image))
    
    
    test_hu_moments(images)
    test_zernike_moments(images)
    test_hog(images)
    #test_sift(images)


def show_features(X_pca, Y, pca: PCA, n_components: int):
    total_var = pca.explained_variance_ratio_.sum() * 100
    
    exp_var_cumul = np.cumsum(pca.explained_variance_ratio_)

    fig = px.area(
        x=range(1, exp_var_cumul.shape[0] + 1),
        y=exp_var_cumul,
        labels={"x": "# Components", "y": "Explained Variance"}
    )
    fig.show()
    
    """fig = px.scatter_matrix(
        X_pca,
        labels=Y,
        dimensions=range(n_components),
        color=Y
    )
    fig.update_traces(diagonal_visible=False)
    fig.show()"""
    
    fig = px.scatter_3d(
        X_pca, x=0, y=1, z=2, 
        labels=Y,
        color=Y,
        title=f'Total Explained Variance: {total_var:.2f}%',
    )
    fig.show()



if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read('config.ini')
    image_dir = config['Paths']['training_dir']
    image_paths = [os.path.join(image_dir, file) for file in os.listdir(image_dir)]
    
    test_all(image_paths)
    