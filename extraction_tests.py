import shared.feature_extraction
import shared.preprocessing
import configparser
import os
import cv2
import numpy as np
import sklearn.preprocessing
import matplotlib.pyplot as plt

from typing import Callable


def test_hu_moments(images):
    X = []  # features
    Y = []  # labels
    
    for image in images:
        shape_name = image.split('_')[-2]
        
        image = shared.preprocessing.read_image(image)
        
        preprocessed_image = shared.preprocessing.preprocess_image(image)
        
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
        
    encoder = sklearn.preprocessing.LabelEncoder()
    Y_le = encoder.fit_transform(Y)
    Y = encoder.inverse_transform(Y_le)
    
    show_features(X, Y)
            

def test_all(images):
    test_hu_moments(images)


def show_features(X, Y):
    label_groups: dict[str, list[np.ndarray]] = {}
    for label, features in zip(Y, X):
        if label not in label_groups:
            label_groups[label] = []
        label_groups[label].append(features)
        
    num_labels = len(label_groups)
    num_rows = int(np.ceil(np.sqrt(num_labels)))
    num_cols = int(np.ceil(num_labels / num_rows))
    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(15, 12))
    axes_flat = axes.ravel()
        
    for idx, (label, features_list) in enumerate(label_groups.items()):
        features_list = np.array(features_list)
        
        ax = axes_flat[idx]
        
        for i in range(features_list.shape[1]):
            ax.scatter(np.arange(features_list.shape[0]), features_list[:, i], label=f'Feature {i}')
            
        ax.set_title(label)
        if (idx == 0):
            ax.legend()
        
    for idx in range(num_labels, num_rows * num_cols):
        axes_flat[idx].axis('off')
        
    plt.tight_layout()
    plt.show()



if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read('config.ini')
    image_dir = config['Paths']['training_dir']
    image_files = [os.path.join(image_dir, file) for file in os.listdir(image_dir)]
    
    test_all(image_files)
    