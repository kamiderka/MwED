import numpy as np
from collections import Counter

from modules.decision_tree import DecisionTree

class RandomForestClassifier:
    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2, max_features="sqrt"):
        self.n_estimators = n_estimators  # Number of trees in the forest
        self.max_depth = max_depth  # Max depth of each tree
        self.min_samples_split = min_samples_split  # Minimum samples required to split
        self.max_features = max_features  # Max features to consider for each split
        self.trees = []  # Container for the decision trees

    def fit(self, X, y):
        self.trees = []  # Clear any previously trained trees
        n_samples = X.shape[0]

        for _ in range(self.n_estimators):
            # Bootstrap sample
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            X_sample = X[indices]
            y_sample = y[indices]

            # Create and train a new decision tree
            tree = DecisionTree(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X):
        # Get predictions from each tree
        tree_predictions = np.array([tree.predict(X) for tree in self.trees])
        # Majority vote
        return np.apply_along_axis(self._most_common_label, axis=0, arr=tree_predictions)

    def _most_common_label(self, y):
        counter = Counter(y)
        return counter.most_common(1)[0][0]


