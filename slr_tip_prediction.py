import numpy as np
import matplotlib.pyplot as plt

# Set plot size
plt.rcParams['figure.figsize'] = (8.0, 5.0)

# Dataset
X = np.array([34, 108, 64, 88, 99, 51])
Y = np.array([5, 17, 11, 8, 14, 5])

# Plot initial data
plt.scatter(X, Y)
plt.title("Scatter Plot of X vs Y")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()

# Initialize
theta1_list = []
theta0_list = []
cost_list = []

theta1 = 0
theta0 = 0

# Learning rate and number of epochs
L = 0.00001
epochs = 800000

# Number of training examples
m = float(len(X))
iteration = []

# Gradient Descent
print('theta0\t, theta1\t, cost\t\tEpoch')
for i in range(epochs):
    # Calculate predicted Y
    Y_pred = theta0 + theta1 * X
    
    # Compute gradients
    temp0 = (2 / m) * sum(Y_pred - Y)
    temp1 = (2 / m) * sum(X * (Y_pred - Y))

    # Update parameters
    theta0 -= L * temp0
    theta1 -= L * temp1

    # Store values for plotting later
    theta0_list.append(theta0)
    theta1_list.append(theta1)

    # Calculate cost
    cost = np.sum((Y_pred - Y)**2)
    cost_list.append(cost)

    # Log and plot every 10 epochs
    if i < 10 or i % (epochs // 100) == 0:
        print(f"{theta0:.6f}\t {theta1:.6f}\t {cost:.6f}\t {i}")
        plt.scatter(X, Y)
        plt.plot([min(X), max(X)], [min(Y_pred), max(Y_pred)], color='red')
        plt.title(f"Iteration {i}")
        plt.xlabel("X")
        plt.ylabel("Y_pred")
        plt.show()

    # Track iteration for cost plotting
    iteration.append(i)

# Final model output
print(f"Final theta0: {theta0:.6f}, theta1: {theta1:.6f}")
plt.scatter(X, Y)
plt.plot([min(X), max(X)], [min(Y_pred), max(Y_pred)], color='red')
plt.title("Final Linear Regression Fit")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()

# Linear Regression Equation
print(f'Linear Regression Equation: Y_pred = {theta0:.6f} + {theta1:.6f}X')

# Cost and R-squared calculations
Y_mean = np.mean(Y)
SST = np.sum((Y - Y_mean) ** 2)  # Total sum of squares
SSE = np.sum((Y - Y_pred) ** 2)  # Sum of squared errors
SSR = SST - SSE  # Regression sum of squares
R_square = (SSR / SST) * 100  # R-squared

# Print performance metrics
print(f"\nSST: {SST:.6f}\nSSR: {SSR:.6f}\nSSE: {SSE:.6f}\nR_Square: {R_square:.2f}%")
