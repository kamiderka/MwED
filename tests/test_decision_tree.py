import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier as SklearnDecisionTree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris, fetch_openml
import sys
import os

import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier as SklearnDecisionTree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris, fetch_openml
import sys
import os

# Ensure module paths are available for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from modules.decision_tree import DecisionTree
from utils.utils import evaluate_models

# Load Iris dataset
iris = load_iris()
X_iris, y_iris = iris.data, iris.target

# Load or handle MNIST dataset
try:
    mnist = fetch_openml("mnist_784", version=1, as_frame=False)  # Specify as_frame=False for NumPy arrays
    # Sample 10,000 instances from MNIST for speed
    X_mnist, y_mnist = mnist.data[:10000], mnist.target[:10000].astype(np.int32)
except Exception as e:
    print(f"Error loading MNIST data: {e}")
    X_mnist, y_mnist = None, None

# Reduced Parameters to Test
max_depth_values = [3, 5]  # Limit to two depths
min_samples_split_values = [2, 10]  # Limit to two splits

# Run evaluations on Iris dataset
iris_custom, iris_sklearn = evaluate_models(X_iris, y_iris, max_depth_values, min_samples_split_values)

# Check if MNIST data was loaded successfully before evaluation
if X_mnist is not None and y_mnist is not None:
    mnist_custom, mnist_sklearn = evaluate_models(X_mnist, y_mnist, max_depth_values, min_samples_split_values)
else:
    mnist_custom, mnist_sklearn = None, None

# Plot both datasets in a single window with subplots
fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

# Define colors and markers, ensuring sufficient unique options
colors = ['blue', 'orange']
markers = ['o', 'x']

# Plot Iris dataset results
for i, max_depth in enumerate(max_depth_values):
    axes[0].plot(min_samples_split_values, iris_custom[i], label=f'Custom Tree, max_depth={max_depth}', 
                 marker=markers[0], color=colors[i % len(colors)])
    axes[0].plot(min_samples_split_values, iris_sklearn[i], label=f'Sklearn Tree, max_depth={max_depth}', 
                 marker=markers[1], linestyle='--', color=colors[i % len(colors)])

axes[0].set_title('Iris Dataset')
axes[0].set_xlabel('min_samples_split')
axes[0].set_ylabel('Accuracy')
axes[0].grid(True)

# Plot MNIST dataset results if available
if mnist_custom is not None and mnist_sklearn is not None:
    for i, max_depth in enumerate(max_depth_values):
        axes[1].plot(min_samples_split_values, mnist_custom[i], label=f'Custom Tree, max_depth={max_depth}', 
                     marker=markers[0], color=colors[i % len(colors)])
        axes[1].plot(min_samples_split_values, mnist_sklearn[i], label=f'Sklearn Tree, max_depth={max_depth}', 
                     marker=markers[1], linestyle='--', color=colors[i % len(colors)])

    axes[1].set_title('MNIST Dataset')
else:
    axes[1].text(0.5, 0.5, 'MNIST Data Not Loaded', ha='center', va='center', fontsize=12)
axes[1].set_xlabel('min_samples_split')
axes[1].grid(True)

# Add a single legend for both plots
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=len(max_depth_values), title="Model and max_depth")

plt.suptitle('Comparison of Custom and Sklearn Decision Trees on Iris and MNIST Datasets')
plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to make room for the main title
plt.show()

