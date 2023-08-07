import sklearn.preprocessing
import numpy as np

from typing import Tuple

def scale_data(X, Y) -> Tuple[np.array, np.array]:
    scaler = sklearn.preprocessing.StandardScaler()
    X_np = np.array(X)
    X_scaled = scaler.fit_transform(X_np)
        
    encoder = sklearn.preprocessing.LabelEncoder()
    Y_np = np.array(Y)
    Y_encoded = encoder.fit_transform(Y_np)
    Y_decoded = encoder.inverse_transform(Y_encoded)
    
    return X_scaled, Y_decoded


