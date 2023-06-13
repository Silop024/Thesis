import os
import configparser
from humoments_svm_classifier import ShapeClassifier

def main():
    # Initialize
    config = get_config()
    
    rejecter_path = f"{config['model_dir']}humoments_one_svm_model.pkl"
    classifier_path = f"{config['model_dir']}humoments_multi_svm_model.pkl"
    
    rejecter = ShapeClassifier(model_path=rejecter_path)
    classifier = ShapeClassifier(model_path=classifier_path)
    
    # Get a list of all the image files in the directory
    image_files = os.listdir(config['image_dir'])
    
    # Classify shapes in each image
    for image_file in image_files:
        image_path = f"{config['image_dir']}{image_file}"
        rejecter.classifyShapes(image_path)
        classifier.classifyShapes(image_path)
        
        filter_classifiers(classifier=classifier, rejecter=rejecter)
            
        classifier.showPredictions(image_path)
        rejecter.showPredictions(image_path)
    
    
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
    
    
def get_config() -> dict[str, str]:
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