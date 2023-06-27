import os
import sys
import ast
import argparse
import configparser
import joblib

# My modules
from training.svm_training import Trainer
from shared.debugging import Debug
from svm.humoments_classifier import ShapeClassifier


def main(args: dict[str, any]):
    # Read configs
    config = configparser.ConfigParser()
    config.read('config.ini') 
    model_dir = config['Paths']['model_dir']
    models = config['Models']
    
    # Read args
    image_path: str = args['image']
    method: str = args['method']
    verbosity: int = args['verbose']
    train: bool = args['train']
    name: str = args['name']
    pca_n: int = args['pca']
    
    
    
    Debug.set_verbosity(verbosity)
    
    if train:
        trainer = Trainer()
        params_in = args['params']
        params = None if params_in is None else ast.literal_eval(params_in)
        
        estimator_name = args['estimator']
        estimator = None
        for clf_class, (clf_name, _) in trainer.classifiers.items():
            if clf_name == estimator_name:
                estimator = clf_class()
        
        
        estimator, pca = trainer.train(
            train_dir=config['Paths']['training_dir'],
            pca_n=pca_n,
            exhaustive=args['exhaustive'],
            estimator=estimator,
            params=params
        )
        joblib.dump(estimator, os.path.join(model_dir, f"{name}.joblib"))
        if pca_n > 0:
            joblib.dump(pca, os.path.join(model_dir, 'pca.joblib'))
        
        if not args['run']:
            return
    
    pca_path = f"{model_dir}{'pca.joblib'}" if pca_n > 0 else None
                
    match method:
        case 'svm_with_clustering':
            classifier_path = f'{model_dir}{name}.joblib'
            
        case _:
            print('Unknown method parameter')
            return
        
    classifier = ShapeClassifier(model_path=classifier_path, pca_path=pca_path)
    classifier.classifyShapes(image_path)
        
    classifier.showPredictions()
    
    
def train_promter():
    pass


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
                        help='set verbosity of the output [0-4] (default: %(default)s)')
    parser.add_argument('-n', '--name', default='model', 
                        help='the name of the model to used for classification')
    parser.add_argument('-pca', type=int, default=0,
                        help='specify whether to use pca or not, and if so, how many features')

    subparsers = parser.add_subparsers()
    
    parser_train = subparsers.add_parser('train', help='sub-command for training models')
    # Set a flag called 'train' that will be True iff the 'train' subcommand is used
    parser.set_defaults(train=False)
    parser_train.set_defaults(train=True)
    
    parser_train.add_argument('-e', '--estimator', default=None, 
                              help='specify what type of estimator to train (default: %(default)s)',
                              choices=[
                                  None,
                                  'Logistic Regression', 
                                  'Random Forest',
                                  'Gradient Boosting',
                                  'SVC',
                                  'K-Neighbors',
                                  'Decision Tree',
                                  'Naive Bayes'
                               ]
    )
    parser_train.add_argument('-p', '--params', default=None,
                              help='hyperparameters used for training the estimator (default: %(default)s)')
    parser_train.add_argument('-x', '--exhaustive', action='store_true',
                              help='use an exhaustive search, takes longer (default %(default)s)')
    parser_train.add_argument('-r', '--run', action='store_true',
                              help='run after training (deafult: %(default)s)')
    parser_train.add_argument('-n', '--name', default='model', 
                              help='the name of the model to be trained')


    # Get args and run
    args = vars(parser.parse_args(sys.argv[1:]))
    main(args)