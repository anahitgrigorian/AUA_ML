import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

#generating non-linear noisy data
X, y = make_moons(n_samples=500, noise=0.2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#plotting the decision boundary
def plot_decision_boundary(model, X, y, title, step=0.1):   
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, step), np.arange(y_min, y_max, step))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k')
    plt.title(title)
    plt.show()

#Hard Margin SVM and Soft Margin SVM
svm_hard = SVC(kernel='linear', C=1e10)
svm_hard.fit(X_train, y_train)
plot_decision_boundary(svm_hard, X_test, y_test, 'Hard Margin SVM', step=0.2)

svm_soft = SVC(kernel='rbf', C=1)
svm_soft.fit(X_train, y_train)
plot_decision_boundary(svm_soft, X_test, y_test, 'Soft Margin SVM', step=0.2)
