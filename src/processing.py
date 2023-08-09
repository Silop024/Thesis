# Installed with pip
import numpy as np
import sklearn.preprocessing
from sklearn.decomposition import PCA


def scale_data(X: np.ndarray) -> np.ndarray:
    scaler = sklearn.preprocessing.StandardScaler()
    X_np = np.array(X)
    X_scaled = scaler.fit_transform(X_np)
    
    return X_scaled


def fix_labels(Y: np.array) -> np.array:
    encoder = sklearn.preprocessing.LabelEncoder()
    Y_np = np.array(Y)
    Y_encoded = encoder.fit_transform(Y_np)
    Y_decoded = encoder.inverse_transform(Y_encoded)
    
    return Y_decoded


def create_pca(X: np.ndarray, n_components: int) -> PCA:
    pca = PCA(n_components=n_components, random_state=0)
    
    pca.fit(X)
    
    return pca


def use_pca(X:np.ndarray, pca: PCA) -> np.ndarray:
    return pca.transform(X)