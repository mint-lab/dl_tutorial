import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

midterm_range = np.array([0, 125])
final_range = np.array([0, 100])

# Load score data
class_kr = np.loadtxt('data/class_score_kr.csv', delimiter=',')
class_en = np.loadtxt('data/class_score_en.csv', delimiter=',')
data = np.vstack((class_kr, class_en))
x = data[:,0].reshape(-1, 1)
y = data[:,1]

# Instantiate regression models
models = [
    {'name': 'svm.SVR(poly,1)', 'obj': svm.SVR(kernel='poly', degree=1, coef0=1), 'color': 'g'},
    {'name': 'svm.SVR(poly,5)', 'obj': svm.SVR(kernel='poly', degree=5, coef0=1), 'color': 'b'},
]

# Prepare for plotting
plt.figure()
plt.plot(data[:,0], data[:,1], 'r.', label='The given data')
idx_sort = np.argsort(data[:,0])

for model in models:
    # Train a model
    model['obj'].fit(x, y)

    # Visualize regression results (the estimated line)
    y_pred = model['obj'].predict(x)
    model['mae'] = np.mean(np.abs(y_pred - y))
    plt.plot(x[idx_sort,0], y_pred[idx_sort], label=f'{model["name"]} (MAE={model["mae"]:.1f})', color=model['color'])

# Decorate the plot
plt.xlabel('Midterm scores')
plt.ylabel('Final scores')
plt.xlim(midterm_range)
plt.ylim(final_range)
plt.grid()
plt.legend()
plt.show()