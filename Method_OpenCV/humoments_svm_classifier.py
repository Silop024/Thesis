import joblib
import cv2
import numpy as np
from preprocessing import read_image_and_scale
from processing import read_image_and_process
from shapes import ShapePrediction

class ShapeClassifier:
    def __init__(self, model_path: str):
        self.clf = joblib.load(model_path)
        self.predictions: list[ShapePrediction]

    def classifyShapes(self, image_path: str, debug: bool = False):
        processed_image = read_image_and_process(image_path, debug)
        self.predictions = []
        
        X = []
        for huMoment in processed_image['humoments']:
            # Flatten the Hu Moments arrays and add it to the list of feature vectors
            X.append(huMoment.flatten())
        X = np.array(X)
        classifications = self.clf.predict(X)
        
        # Only useful for the one class classifier that acts as a rejecter.
        for i, centroid in enumerate(processed_image['centroids']):
            if classifications[i] == -1:
                self.predictions.append(
                    ShapePrediction(
                        index = i, 
                        position = centroid, 
                        is_keyword = False, 
                        keyword = '', 
                        id = 0, 
                        image_path = image_path
                    )
                )
            else:
                self.predictions.append(
                    ShapePrediction(
                        index = i, 
                        position = centroid, 
                        is_keyword = True, 
                        keyword = str(classifications[i]),
                        id = -1,
                        image_path = image_path
                    )
                )
                
        
    def showPredictions(self, image_path: str):
        if len(self.predictions) < 1:
            return
        
        image = read_image_and_scale(image_path)
        
        for prediction in self.predictions:
            cv2.putText(image, prediction.__repr__(), prediction.position, cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.4, color=(0, 0, 255), thickness=1)
        
        cv2.imshow('Predictions', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
