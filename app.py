import os
import sys
import argparse
import configparser
from training.humoments_svm_training import train
from shared.debugging import Debug
from svm.humoments_classifier import ShapeClassifier


def main(args: dict[str, any]):
    config = configparser.ConfigParser()
    config.read('config.ini')
    
    image_path: str = args['image']
    method: str = args['method']
    verbosity: int = args['verbose']
    
    train_only: bool = args['train_only']
    train_run: bool = args['train_run']
    
    Debug.set_verbosity(verbosity)
    
    if train_only or train_run:
        train()
        if train_only:
            return
    
    match method:
        case 'svm_with_clustering':
            print('classifying with svm and clustering')
            classifier_path = f"{config['Paths']['model_dir']}{config['Models']['calibrated_classifier']}"
            classifier = ShapeClassifier(model_path=classifier_path)
            classifier.classifyShapes(image_path, verbosity > 3)
        
            classifier.showPredictions()
            return
        case _:
            print('Unknown method parameter')
            return
    
    

if __name__ == "__main__":
    # Add local directory to PYTHONPATH
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))


    # Add arguments
    parser = argparse.ArgumentParser(
        prog='app.py',
        description='Converts an image containing certain defined and any undefined shapes into code'
    )
    parser.add_argument('-i', '--image', default='example.jpg',
                    help='the path to the image you wish to convert (default: %(default)s)')
    parser.add_argument('-m', '--method', default='svm_with_clustering',
                    help='the method you wish to use for the conversion (default: %(default)s)',
                    choices=['svm_with_clustering'])
    parser.add_argument('-v', '--verbose', action='count', default=0,
                    help='set verbosity of the output [0-2] (default: %(default)s)')

    mutex_group = parser.add_mutually_exclusive_group(required=False)
    mutex_group.add_argument('-to', '--train-only', action='store_true',
                         help="train only, don't run (default: %(default)s)")
    mutex_group.add_argument('-tr', '--train-run', action='store_true',
                         help='train and run (default: %(default)s)')   
    
    # Get args and run
    args = vars(parser.parse_args(sys.argv[1:]))
    main(args)