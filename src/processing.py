# Installed with pip
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder


def create_scaler(X: np.ndarray) -> StandardScaler:
    scaler = StandardScaler()
    
    X_np = np.array(X)
    
    return scaler.fit(X_np)


def scale_data(X: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    X_np = np.array(X)
    X_scaled = scaler.transform(X_np)
    
    return X_scaled


def create_encoder(Y: np.array) -> LabelEncoder:
    encoder = LabelEncoder()
    
    Y_np = np.array(Y)
    
    return encoder.fit(Y_np)


def fix_labels(Y: np.array, encoder: LabelEncoder) -> np.array:
    Y_np = np.array(Y)
    
    Y_encoded = encoder.transform(Y_np)
    Y_decoded = encoder.inverse_transform(Y_encoded)
    
    return Y_decoded


def create_pca(X: np.ndarray, n_components: int) -> PCA:
    pca = PCA(n_components=n_components, random_state=0)
    
    pca.fit(X)
    
    return pca


def use_pca(X: np.ndarray, pca: PCA) -> np.ndarray:
    return pca.transform(X)