import os
import configparser
from humoments_classifier import ShapeClassifier
from sklearn.cluster import DBSCAN

def main():
    # Initialize
    config = configparser.ConfigParser()
    config.read('config.ini')
    
    classifier_path = f"{config['Paths']['model_dir']}{config['Models']['calibrated_classifier']}"
    
    classifier = ShapeClassifier(model_path=classifier_path)
    clusterer = DBSCAN(eps=0.2, min_samples=2)
    
    # Get a list of all the image files in the test image directory
    image_dir = config['Paths']['test_dir']
    image_files = os.listdir(image_dir)
    
    # Classify shapes in each image
    for image_file in image_files:
        image_path = f"{image_dir}{image_file}"
        print(f'---------- {image_path} ----------')
        
        classifier.classifyShapes(image_path, True)
        
        classifier.showPredictions()
        
        #filter_classifiers(classifier=classifier, rejecter=rejecter)
        
        """cluster_ids = cluster_analysis(image_path, clusterer)
        
        print(cluster_ids)
        
        for i, shape in enumerate(rejecter.predictions):
            if shape.is_keyword:
                continue
            classifier.predictions[i].id = cluster_ids[i]
            classifier.predictions[i].shape = Shape.Undefined
            
        classifier.showPredictions()"""
        #rejecter.showPredictions()
    
    
def filter_classifiers(classifier: ShapeClassifier, rejecter: ShapeClassifier):    
    non_keywords = []
    keywords = []
    for p in range(len(rejecter.predictions)):
        if rejecter.predictions[p].is_keyword:
            keywords.append(rejecter.predictions[p])
        else:
            non_keywords.append(classifier.predictions[p])
                
    for p in non_keywords:
        classifier.predictions.remove(p)

    for p in keywords:
        rejecter.predictions.remove(p)
    
     
    
if __name__ == "__main__":
    main()