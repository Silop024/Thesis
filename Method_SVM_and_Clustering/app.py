import os
import configparser
from processing import cluster_analysis
from humoments_svm_classifier import ShapeClassifier
from sklearn.cluster import DBSCAN
from shapes import Shape

def main():
    # Initialize
    dirs = get_directories()
    
    rejecter_path = f"{dirs['model_dir']}humoments_one_svm_model.pkl"
    classifier_path = f"{dirs['model_dir']}humoments_multi_svm_model.pkl"
    
    rejecter = ShapeClassifier(model_path=rejecter_path)
    classifier = ShapeClassifier(model_path=classifier_path)
    clusterer = DBSCAN(eps=0.2, min_samples=2)
    
    # Get a list of all the image files in the directory
    image_files = os.listdir(dirs['image_dir'])
    
    # Classify shapes in each image
    for image_file in image_files:
        image_path = f"{dirs['image_dir']}{image_file}"
        
        rejecter.classifyShapes(image_path)
        classifier.classifyShapes(image_path)
        
        classifier.showPredictions()
        
        #filter_classifiers(classifier=classifier, rejecter=rejecter)
        
        cluster_ids = cluster_analysis(image_path, clusterer)
        
        print(cluster_ids)
        
        for i, shape in enumerate(rejecter.predictions):
            if shape.is_keyword:
                continue
            classifier.predictions[i].id = cluster_ids[i]
            classifier.predictions[i].shape = Shape.Undefined
            
        classifier.showPredictions()
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
    
    
def get_directories() -> dict[str, str]:
    # Get config parser
    config = configparser.ConfigParser()
    config.read('config.ini')
    
    dirs = {}
    
    # Directory where the trained model are stored
    dirs.update({'model_dir': config.get('Paths', 'model_dir')})

    # Directory where the test images are stored
    dirs.update({'image_dir': config.get('Paths', 'test_dir')})
    
    return dirs
     
    
if __name__ == "__main__":
    main()