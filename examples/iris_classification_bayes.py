import numpy as np
import matplotlib.pyplot as plt
from sklearn import (datasets, naive_bayes, metrics)
from matplotlib.colors import ListedColormap
from scipy.stats import multivariate_normal

# Load a dataset partially
iris = datasets.load_iris()
iris.data = iris.data[:,0:2]
iris.feature_names = iris.feature_names[0:2]
iris.color = np.array([(1, 0, 0), (0, 1, 0), (0, 0, 1)])

# Train a model
model = naive_bayes.GaussianNB()
model.fit(iris.data, iris.target)
#model.class_prior_ = [0.1, 0.6, 0.3] # Try this to give manual prior

# Validate training
for c in range(len(model.classes_)):
    data = iris.data[iris.target == c,:]
    print(f'## Class {c}')
    print('  * Trained prior    = ' + np.array2string(model.class_prior_[c], precision=3))
    print('  * Manual  prior    = ' + '{:.3f}'.format(len(data) / len(iris.data)))
    print('  * Trained mean     = ' + np.array2string(model.theta_[c], precision=3))
    print('  * Manual  mean     = ' + np.array2string(np.mean(data, axis=0), precision=3))
    print('  * Trained variance = ' + np.array2string(model.sigma_[c], precision=3))
    print('  * Manual  variance = ' + np.array2string(np.var(data, axis=0), precision=3))

# Visualize training results (decision boundaries)
x_min, x_max = iris.data[:, 0].min() - 1, iris.data[:, 0].max() + 1
y_min, y_max = iris.data[:, 1].min() - 1, iris.data[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
xy = np.vstack((xx.flatten(), yy.flatten())).T
zz = model.predict(xy)
plt.contourf(xx, yy, zz.reshape(xx.shape), cmap=ListedColormap(iris.color), alpha=0.2)

# Visualize training results (trained Gaussians)
for c in range(len(model.classes_)):
    likelihood = multivariate_normal(model.theta_[c], np.diag(model.sigma_[c]))
    zz = model.class_prior_[c] * likelihood.pdf(xy)
    plt.contour(xx, yy, zz.reshape(xx.shape), cmap=ListedColormap(iris.color[c]), alpha=0.4)

# Test the model
predict = model.predict(iris.data)
accuracy = metrics.balanced_accuracy_score(iris.target, predict)

# Visualize testing results
plt.title(f'naive_bayes.Gaussian ({accuracy:.3f})')
plt.scatter(iris.data[:,0], iris.data[:,1], c=iris.color[iris.target], edgecolors=iris.color[predict])
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.show()
