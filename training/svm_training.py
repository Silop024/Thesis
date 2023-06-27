import os
import numpy as np

from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA

from shared.debugging import Debug
from svm.processing import read_dir_and_process


class Trainer:
    def __init__(self) -> None:
        self.classifiers = {
            LogisticRegression: ('Logistic Regression', {'C': [0.1, 1, 10], 'max_iter': [1000]}),
            RandomForestClassifier: ('Random Forest', {'n_estimators': [10, 50, 100, 200, 300, 400]}),
            GradientBoostingClassifier: ('Gradient Boosting', {'n_estimators': [10, 50, 100, 200, 300, 400], 'learning_rate': [0.01, 0.1, 1]}),
            SVC: ('SVC', {'C': [0.1, 1, 10, 100], 'gamma': [0.01, 0.1, 1, 10],'kernel': ['linear', 'rbf', 'poly']}),
            KNeighborsClassifier: ('K-Neighbors', {'n_neighbors': [3, 5, 7, 9]}),
            DecisionTreeClassifier: ('Decision Tree', {'max_depth': [None, 10, 20, 30, 50]}),
            GaussianNB: ('Naive Bayes', {})
        }
    
    
    def train(self, 
              train_dir: str,
              pca_n: int = 0,
              exhaustive: bool = False, 
              estimator: BaseEstimator = None,
              params: dict = None) -> tuple[BaseEstimator, PCA]:
        """
        Trains an estimator given the features and labels.

        Args:
            features (np.ndarray): The features to train the classifier.
            labels (np.ndarray): The corresponding labels for the features.
            exhaustive (bool, optional): If True, uses GridSearchCV for exhaustive search. If False, uses RandomizedSearchCV.
            estimator (BaseEstimator, optional): The estimator to be trained. If None, the best estimator is found using `find_best_estimator`.
            params (dict): The parameters for the estimator. If None, the best parameters are found using GridSearchCV or RandomizedSearchCV.
            
        Returns:
            BaseEstimator: The trained estimator.
        """
        # Get a list of all the image files in the directory
        features, labels = read_dir_and_process(train_dir)
        
        # Principal component analysis to reduce noise of data
        if pca_n > 0:
            pca = PCA(n_components=pca_n)
            features = pca.fit_transform(features)
        else:
            pca = None
        
        Debug.print('Training model...', level=1, color='red')
        
        if estimator is None:
            estimator, params = self.find_best_estimator(features, labels, exhaustive)
            Debug.print(f'Best estimator found: {estimator.__str__()}: {params}', level=1, color='green')
        elif params is None:
            _, params = self.classifiers[type(estimator)]
            if exhaustive:
                search_cv = GridSearchCV(estimator, params, cv=5)
            else:
                search_cv = RandomizedSearchCV(estimator, params, n_iter=10, cv=5, random_state=0)
            search_cv.fit(features, labels)
            estimator = search_cv.best_estimator_

            Debug.print(f'Best params found: {search_cv.best_params_}', level=1, color='green')
        else:
            estimator.set_params(**params)
            
        estimator.fit(features, labels)
        
        Debug.print('Training model... Done', level=1, color='green')

        return estimator, pca
    
    
    def find_best_estimator(self, 
                            features: np.ndarray, 
                            labels: np.ndarray, 
                            exhaustive: bool) -> tuple[BaseEstimator, dict]:
        """
        Searches for the best estimator given the features and labels.

        Args:
            features (np.ndarray): The features to train the classifier
            labels (np.ndarray): The corresponding labels for the features
            exhaustive (bool): If true, use GridSearchCV for an exhaustive search. If False, uses RandomizedSearchCV

        Returns:
            (BaseEstimator, str): A tuple of the best estimator and its parameters
        """
        Debug.print('Finding best estimator', level=1, color='red')
        
        best_score = 0.0
        best_estimator: BaseEstimator = None
        best_params: dict = None
        
        for clf_class, (name, param_grid) in self.classifiers.items():
            # Calculate the total number of combinations
            n_combinations = np.prod([len(v) if isinstance(v, list) else 1 for v in param_grid.values()])
            n_iter = min(n_combinations, 10)
            
            clf = clf_class()  # instantiate classifier
            if exhaustive:
                search_cv = GridSearchCV(clf, param_grid, cv=5)
            else:
                search_cv = RandomizedSearchCV(clf, param_grid, n_iter=n_iter, cv=5, random_state=0)
            search_cv.fit(features, labels)
            
            if search_cv.best_score_ > best_score:
                best_score = search_cv.best_score_
                best_estimator = search_cv.best_estimator_
                best_params = search_cv.best_params_
            
            Debug.print(f'{name}: Best Parameters: {search_cv.best_params_} Best Score: {search_cv.best_score_:.2f}', level=2, color='blue')
                
        return best_estimator, best_params
                
            