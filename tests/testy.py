import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from decision_tree import DecisionTree

iris = datasets.load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Nasze drzewo
my_tree = DecisionTree(max_depth=3)
my_tree.fit(X_train, y_train)
my_predictions = my_tree.predict(X_test)

# Wzorcowe drzewo
sklearn_tree = DecisionTreeClassifier(max_depth=3, random_state=42)
sklearn_tree.fit(X_train, y_train)
sklearn_predictions = sklearn_tree.predict(X_test)

print("Wyniki z własnego drzewa:")
print(my_predictions)
print("Wyniki z scikit-learn:")
print(sklearn_predictions)

my_accuracy = np.mean(my_predictions == y_test)
sklearn_accuracy = np.mean(sklearn_predictions == y_test)

print(f"\nDokładność własnego drzewa: {my_accuracy:.2f}")
print(f"Dokładność drzewa scikit-learn: {sklearn_accuracy:.2f}")
