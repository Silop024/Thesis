import joblib
import cv2
import numpy as np
from sklearn.svm import SVC
from preprocessing import read_image_and_scale
from processing import read_image_and_process
from shapes import ShapePrediction, Shape

class ShapeClassifier:
    clf: SVC
    predictions: list[ShapePrediction]
    source: str
    
    def __init__(self, model_path: str):
        self.clf = joblib.load(model_path)
        self.predictions = []
        self.source = ''

    def classifyShapes(self, image_path: str, debug: bool = False):
        """
        Classifies the shapes present in the given image.
        
        This method first reads and processes the image at the specified path, then extracts Hu moments 
        and centroids for each detected shape. These feature vectors are then classified using the SVM model.
        
        The results of the classification are stored in the `predictions` attribute as a list of `ShapePrediction` instances.

        Args:
            image_path (str): The path to the image where shapes are to be classified.
            debug (bool, optional): If True, additional debug information will be displayed. Defaults to False.
        """        
        processed_image = read_image_and_process(image_path, debug)
        self.predictions = []
        self.source = image_path
        
        classifications = self.clf.predict(processed_image['humoments'])
        
        # Only useful for the one class classifier that acts as a rejecter.
        for i, centroid in enumerate(processed_image['centroids']):
            self.predictions.append(
                ShapePrediction(
                    index = i, 
                    position = centroid, 
                    shape = Shape(str(classifications[i])),
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
        
        for prediction in self.predictions:
            print(prediction)
            
        cv2.imshow('Predictions', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
