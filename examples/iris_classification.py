import numpy as np
import matplotlib.pyplot as plt
from sklearn import (datasets, svm)
from matplotlib.lines import Line2D # For the custom legend

# Load a dataset
iris = datasets.load_iris()

# Train a model
model = svm.SVC()                   # Accuracy: 0.973 (146/150)
model.fit(iris.data, iris.target)   # Try 'iris.data[:,0:2]' (Accuracy: 0.820)

# Test the model
predict = model.predict(iris.data)  # Try 'iris.data[:,0:2]' (Accuracy: 0.820)
n_correct = sum(predict == iris.target)
accuracy = n_correct / len(iris.data)

# Visualize testing results
cmap = np.array([(1, 0, 0), (0, 1, 0), (0, 0, 1)])
clabel = [Line2D([0], [0], marker='o', lw=0, label=iris.target_names[i], color=cmap[i]) for i in range(len(cmap))]
for (x, y) in [(0, 1), (2, 3)]:
    plt.figure ()
    plt.title(f'svm.SVC ({n_correct}/{len(iris.data)}={accuracy:.3f})')
    plt.scatter(iris.data[:,x], iris.data[:,y], c=cmap[iris.target], edgecolors=cmap[predict])
    plt.xlabel(iris.feature_names[x])
    plt.ylabel(iris.feature_names[y])
    plt.legend(handles=clabel, framealpha=0.5)
plt.show()