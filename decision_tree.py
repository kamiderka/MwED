import numpy as np
import random

class DecisionTree:
    def __init__(self, max_depth=None, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def fit(self, X, y):
        self.tree = self._build_tree(X, y)

    def _build_tree(self, X, y, depth=0):
        num_samples, num_features = X.shape
        unique_classes = np.unique(y)

        if (len(unique_classes) == 1 or
                num_samples < self.min_samples_split or
                (self.max_depth and depth >= self.max_depth)):
            leaf_value = self._most_common_label(y)
            return LeafNode(leaf_value)

        best_split = self._best_split(X, y, num_features)
        if best_split is None:
            leaf_value = self._most_common_label(y)
            return LeafNode(leaf_value)

        left_indices, right_indices = best_split['indices_left'], best_split['indices_right']
        left_subtree = self._build_tree(X[left_indices], y[left_indices], depth + 1)
        right_subtree = self._build_tree(X[right_indices], y[right_indices], depth + 1)

        return DecisionNode(best_split['feature_index'], best_split['threshold'], left_subtree, right_subtree)



    def _best_split(self, X, y, num_features):
        best_gain = -1
        best_split = None

        # Randomly sample sqrt(num_features) features
        feature_indices = random.sample(range(num_features), k=int(np.sqrt(num_features)))

        for feature_index in feature_indices:
            thresholds = np.unique(X[:, feature_index])
            for threshold in thresholds:
                indices_left = np.where(X[:, feature_index] <= threshold)[0]
                indices_right = np.where(X[:, feature_index] > threshold)[0]

                if len(indices_left) == 0 or len(indices_right) == 0:
                    continue

                gain = self._information_gain(y, indices_left, indices_right)

                if gain > best_gain:
                    best_gain = gain
                    best_split = {
                        'feature_index': feature_index,
                        'threshold': threshold,
                        'indices_left': indices_left,
                        'indices_right': indices_right
                    }

        return best_split

    def _information_gain(self, y, indices_left, indices_right):
        p = float(len(indices_left)) / len(y)
        return self._gini_impurity(y) - p * self._gini_impurity(y[indices_left]) - (1 - p) * self._gini_impurity(
            y[indices_right])

    def _gini_impurity(self, y):
        classes, counts = np.unique(y, return_counts=True)
        impurity = 1 - sum((counts / counts.sum()) ** 2)
        return impurity

    def _most_common_label(self, y):
        return np.bincount(y).argmax()

    # Przewidywanie wartości ze zbudowanego drzewa
    def predict(self, X):
        return np.array([self._predict_sample(sample, self.tree) for sample in X])

    def _predict_sample(self, sample, tree):
        if isinstance(tree, LeafNode):
            return tree.value
        if sample[tree.feature_index] <= tree.threshold:
            return self._predict_sample(sample, tree.left)
        else:
            return self._predict_sample(sample, tree.right)


class LeafNode:
    def __init__(self, value):
        self.value = value


class DecisionNode:
    def __init__(self, feature_index, threshold, left, right):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right