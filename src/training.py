# My own files
import processing
import preprocessing
from features import FeatureType
from feature_extraction import extract_all_features

# Python standard libraries
import os
import configparser
from typing import List, Tuple

# Installed with pip
import joblib
import numpy as np
from sklearn import svm
import plotly.express as px
from sklearn.decomposition import PCA


def train(images: List[Tuple[str, np.ndarray]]):
    X = []  # features
    Y = []  # labels
    
    # Get all features from the image
    for image_tuple in images:
        x, y = extract_all_features(image_tuple, FeatureType.HOG)
        
        X.extend(x)
        Y.extend(y)
        
    # Process data before training
    X = processing.scale_data(X)
    Y = processing.fix_labels(Y)
    
    pca = processing.create_pca(X, n_components=5) # Fit a principal component analyser (pca)
    X = processing.use_pca(X, pca) # Use the pca to transform the data to the desire dimensionality.
    
    # Train classifier
    clf = svm.SVC()
    clf.fit(X, Y)
    
    # Save the classifier and pca, the same pca needs to be used with the same classifier
    # for every classification task. Otherwise it will most likely produce an error.
    model_dir = config['Paths']['model_dir']
    
    joblib.dump(pca, os.path.join(model_dir, 'pca.joblib'))
    joblib.dump(clf, os.path.join(model_dir, 'clf.joblib'))
    
    
def show_features(X_pca, Y, pca: PCA):
    exp_var_cumul = np.cumsum(pca.explained_variance_ratio_)
    total_var = pca.explained_variance_ratio_.sum() * 100
        
    # Cumulative explained variance ratio chart
    fig = px.area(
        x=range(1, exp_var_cumul.shape[0] + 1),
        y=exp_var_cumul,
        labels={"x": "# Components", "y": "Explained Variance"},
        title=f'Total Explained Variance: {total_var:.2f}%',
    )
    fig.show()
    
    # Scatter matrix of the principal components
    fig = px.scatter_matrix(
        X_pca,
        dimensions=range(pca.n_components_),
        color=Y
    )
    fig.update_traces(diagonal_visible=False, showupperhalf=False)
    fig.show()
    
    # 3D scatter plot of the 3 first principal components
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
    
    images = []
    for image_path in image_paths:
        image = preprocessing.read_image(image_path)
        preprocessed_image = preprocessing.preprocess_image(image)
        
        file_name = os.path.basename(image_path)
        shape_name = file_name.split('_')[0]
        
        images.append((shape_name, preprocessed_image))
        
    train(images)