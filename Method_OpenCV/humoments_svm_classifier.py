import os
import joblib
import cv2
import numpy as np
import preprocessing
from processing import read_image_and_process
from prediction import ShapePrediction

class ShapeClassifier:
    def __init__(self, model_path: str):
        self.clf = joblib.load(model_path)

    def classifyShapes(self, image_path: str, debug: bool = False) -> list[ShapePrediction]:
        processed_image = read_image_and_process(image_path, debug)
        
        X = []
        for huMoment in processed_image['humoments']:
            # Flatten the Hu Moments arrays and add it to the list of feature vectors
            X.append(huMoment.flatten())
        X = np.array(X)
        classifications = self.clf.predict(X)
        
        predictions = []
        for i, centroid in enumerate(processed_image['centroids']):
            if classifications[i] == -1:
                predictions.append(ShapePrediction(i, centroid, False, "", 0))
            else:
                predictions.append(ShapePrediction(i, centroid, True, classifications[i], 0))
        
        for prediction in predictions:
            print(prediction)
            
        #self.showPredictions(image_path, predictions)
        
        return predictions
        
    def showPredictions(self, image_path: str, predictions: list[ShapePrediction]):
        image = preprocessing.read_image_and_scale(image_path)
        
        for prediction in predictions:
            cv2.putText(image, f'{prediction.index}: {prediction.keyword}', prediction.position, cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.4, color=(0, 0, 255), thickness=1)
        
        cv2.imshow('Predictions', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # Directory where the trained model will be stored
    model_dir = 'C:\\Users\\Jack\\Desktop\\Thesis\\Trained_Models\\'

    # Directory where the training images are stored
    image_dir = 'C:\\Users\\Jack\\Desktop\\Thesis\\Test_Images\\'
    
    rejecter = ShapeClassifier(model_path=f'{model_dir}humoments_one_svm_model.pkl')
    
    classifier = ShapeClassifier(model_path=f'{model_dir}humoments_multi_svm_model.pkl')
    # Get a list of all the image files in the directory
    image_files = os.listdir(image_dir)
    
    # Process each image file
    for image_file in image_files:
        image_path = f'{image_dir}{image_file}'
        predictions_rejecter = rejecter.classifyShapes(image_path)
        predictions_classifier = classifier.classifyShapes(image_path)

        to_remove = []
        for i in range(len(predictions_rejecter)):
            if predictions_rejecter[i].is_keyword == False:
                to_remove.append(predictions_classifier[i])
        
        for i in to_remove:
            predictions_classifier.remove(i)
            
        classifier.showPredictions(f'{image_dir}{image_file}', predictions_classifier)
        