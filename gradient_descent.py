import numpy as np

# Sample data
X = np.array([34, 108, 64, 88, 99, 51, 75, 89, 112, 15, 254, 358])
y = np.array([5, 17, 11, 8, 14, 5, 5, 10, 20, 1, 20, 25])

# Feature scaling (Normalization)
X_mean = np.mean(X)
X_std = np.std(X)
X_scaled = (X - X_mean) / X_std

# Add a bias (ones column) to X
X_b = np.c_[np.ones((len(X_scaled), 1)), X_scaled]

# Initialize theta
theta = np.random.randn(2, 1)

# Hyperparameters
alpha = 0.001  # Reduced learning rate
iterations = 1000
m = len(y)  # Number of samples

# Cost function
def compute_cost(X, y, theta):
    m = len(y)
    predictions = X.dot(theta)
    cost = (1 / (2 * m)) * np.sum(np.square(predictions - y))
    return cost

# Gradient descent algorithm
def gradient_descent(X, y, theta, alpha, iterations):
    m = len(y)
    for i in range(iterations):
        gradients = (1 / m) * X.T.dot(X.dot(theta) - y.reshape(-1, 1))
        theta = theta - alpha * gradients
    return theta

# Running gradient descent
theta_best = gradient_descent(X_b, y, theta, alpha, iterations)

# Prediction with scaled input (we scale the new data too)
X_new = np.array([[200]])  # Example new bill
X_new_scaled = (X_new - X_mean) / X_std
X_new_b = np.c_[np.ones((1, 1)), X_new_scaled]
prediction = X_new_b.dot(theta_best)

print(f"Predicted Tip: {prediction[0][0]:.2f}")
