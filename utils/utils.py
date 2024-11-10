
from sklearn.tree import DecisionTreeClassifier as SklearnDecisionTree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modules.decision_tree import DecisionTree 

# Assuming `DecisionTree` and supporting classes are already defined here
def evaluate_models(X, y, max_depth_values, min_samples_split_values):
    # Splitting data into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    custom_tree_accuracies = []
    sklearn_tree_accuracies = []

    # Test different max_depth and min_samples_split values
    for max_depth in max_depth_values:
        custom_tree_results = []
        sklearn_tree_results = []
        
        for min_samples_split in min_samples_split_values:
            # Custom DecisionTree
            custom_tree = DecisionTree(max_depth=max_depth, min_samples_split=min_samples_split)
            custom_tree.fit(X_train, y_train)
            custom_preds = custom_tree.predict(X_test)
            custom_tree_results.append(accuracy_score(y_test, custom_preds))

            # Scikit-learn DecisionTreeClassifier
            sklearn_tree = SklearnDecisionTree(max_depth=max_depth, min_samples_split=min_samples_split, random_state=42)
            sklearn_tree.fit(X_train, y_train)
            sklearn_preds = sklearn_tree.predict(X_test)
            sklearn_tree_results.append(accuracy_score(y_test, sklearn_preds))
        
        custom_tree_accuracies.append(custom_tree_results)
        sklearn_tree_accuracies.append(sklearn_tree_results)

    return custom_tree_accuracies, sklearn_tree_accuracies

