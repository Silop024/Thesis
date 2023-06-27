import joblib
import cv2
import numpy as np

from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator

# My libraries
from shared.preprocessing import read_image_and_scale
from .processing import read_image_and_process
from .shapes import ShapePrediction, Shape
from shared.debugging import Debug

class ShapeClassifier:
    clf: BaseEstimator
    pca: PCA | None
    predictions: list[ShapePrediction]
    source: str
    
    def __init__(self, model_path: str, pca_path: str | None):
        self.clf = joblib.load(model_path)
        self.pca = None if pca_path is None else joblib.load(pca_path)
        self.predictions = []
        self.source = ''

    def classifyShapes(self, image_path: str):
        """
        Classifies the shapes present in the given image.
        
        This method first reads and processes the image at the specified path, then extracts Hu moments 
        and centroids for each detected shape. These feature vectors are then classified using the SVM model.
        
        The results of the classification are stored in the `predictions` attribute as a list of `ShapePrediction` instances.

        Args:
            image_path (str): The path to the image where shapes are to be classified.
            debug (bool, optional): If True, additional debug information will be displayed. Defaults to False.
        """        
        processed_image = read_image_and_process(image_path)
        self.predictions = []
        self.source = image_path
        
        X = processed_image['humoments']
        
        if self.pca is not None:
            X = self.pca.transform(X)
        
        #classifications = self.clf.predict(X)
        
        # Use the calibrated classifier to predict probabilities
        
        """try:
            probabilities = self.calibrated_clf.predict_proba(X)
        
            # Get the classifications by selecting the class with the highest probability
            max_probabilities = probabilities.max(axis=1)
            classifications = probabilities.argmax(axis=1)
        
            # Only useful for the one class classifier that acts as a rejecter.
            for i, centroid in enumerate(processed_image['centroids']):
                shape_name = self.calibrated_clf.classes_[classifications[i]]
                if max_probabilities[i] < 0.5:
                    shape = Shape.Undefined
                else:
                    shape = Shape(str(shape_name))
                
                self.predictions.append(
                    ShapePrediction(
                        index = i, 
                        position = centroid, 
                        shape = shape,
                        probability = max_probabilities[i],
                        id = 0, 
                        image_path = image_path
                    )
                )
        except:"""
        predictions = self.clf.predict(X)
        for i, centroid in enumerate(processed_image['centroids']):
            self.predictions.append(
                ShapePrediction(
                    index = i, 
                    position = centroid, 
                    shape = Shape(predictions[i]),
                    probability = 1,
                    id = 0, 
                    image_path = image_path
                )
            )
        
                
        
    def showPredictions(self):
        """
        Displays the image where the shapes were classified, with the prediction results overlaid.

        The image is displayed in a new window. The prediction results are displayed as text on top of the 
        image, at the position of the shape that was classified.
        
        Args:
            None
        """
        if len(self.predictions) < 1:
            return
        
        image = read_image_and_scale(self.source)
        
        for prediction in self.predictions:
            cv2.putText(image, prediction.__repr__(), prediction.position, cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.4, color=(0, 0, 255), thickness=1)
        
        if Debug.verbosity > 3:
            for prediction in self.predictions:
                print(prediction)
            
        cv2.imshow('Predictions', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
