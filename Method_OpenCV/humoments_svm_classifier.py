import os
import joblib
import cv2
from cv2 import Mat
from sklearn import svm
from humoments_extractor import getHuMoments
from preprocessor import preprocess

class ShapeClassifier:
    def __init__(self, model_path: str):
        self.clf = joblib.load(model_path)

    def classifyShapes(self, image_path: str, debug: bool = False):
        huMoments, centroids = getHuMoments(image_path, debug)
        
        X = []
        for huMoment in huMoments:
            # Flatten the Hu Moments arrays and add it to the list of feature vectors
            X.append(huMoment.flatten())
        
        y_pred = self.clf.predict(X)
        
        self.showPredictions(image_path=image_path, centroids=centroids, predictions=y_pred)
        
    def showPredictions(self, image_path: str, centroids: list, predictions: list):
        image_copy = preprocess(image_path)
    
        for i, centroid in enumerate(centroids):
            cv2.putText(image_copy, predictions[i], tuple(centroid), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0, 0, 255), thickness=2)
        
        cv2.imshow('Predictions', image_copy)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # Directory where the trained model will be stored
    model_dir = 'C:\\Users\\Jack\\Desktop\\Thesis\\Trained_Models\\'

    # Directory where the training images are stored
    image_dir = 'C:\\Users\\Jack\\Desktop\\Thesis\\Test_Images\\'
    
    
    classifier = ShapeClassifier(model_path=f'{model_dir}humoments_svm_model.pkl')
    # Get a list of all the image files in the directory
    image_files = os.listdir(image_dir)
    
    # Process each image file
    for image_file in image_files:
        classifier.classifyShapes(f'{image_dir}{image_file}', True)
    