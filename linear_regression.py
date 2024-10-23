import numpy as np
import matplotlib.pyplot as plt

# Sample data
X = np.array([34, 108, 64, 88, 99, 51, 75, 89, 112, 15, 254, 358])
y = np.array([5, 17, 11, 8, 14, 5, 5, 10, 20, 1, 20, 25])

# Adding a column of ones for the intercept term (theta_0)
X_b = np.c_[np.ones((len(X), 1)), X]

# Calculate theta using normal equation
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

# Prediction function
def predict(X, theta):
    return X.dot(theta)

# Predictions
X_new = np.array([[0], [400]])
X_new_b = np.c_[np.ones((2, 1)), X_new]
y_predict = predict(X_new_b, theta_best)

# Plotting the results
plt.plot(X_new, y_predict, "r-", label="Predictions")
plt.scatter(X, y)
plt.xlabel("Total Bill")
plt.ylabel("Tip")
plt.title("Simple Linear Regression")
plt.legend()
plt.show()
