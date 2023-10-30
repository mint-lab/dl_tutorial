import numpy as np
import matplotlib.pyplot as plt
from sklearn import (datasets, cluster)
from matplotlib.colors import ListedColormap

# Load a dataset partially
iris = datasets.load_iris()
iris.data = iris.data[:,0:2]                 # Try [:,2:4]
iris.feature_names = iris.feature_names[0:2] # Try [:,2:4]
iris.color = np.array([(1, 0, 0), (0, 1, 0), (0, 0, 1)])

# Train a model
model = cluster.KMeans(n_clusters=3)
model.fit(iris.data)

# Visualize training results (decision boundaries)
x_min, x_max = iris.data[:, 0].min() - 1, iris.data[:, 0].max() + 1
y_min, y_max = iris.data[:, 1].min() - 1, iris.data[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
xy = np.vstack((xx.flatten(), yy.flatten())).T
zz = model.predict(xy)
plt.contourf(xx, yy, zz.reshape(xx.shape), cmap=ListedColormap(iris.color), alpha=0.2)

# Visualize testing results
plt.title('cluster.KMeans')
plt.scatter(iris.data[:,0], iris.data[:,1], c=iris.color[iris.target])
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])

# Visualize training results (mean values)
for c in range(model.n_clusters):
    plt.scatter(*model.cluster_centers_[c], marker='+', s=200, color='k')
plt.show()