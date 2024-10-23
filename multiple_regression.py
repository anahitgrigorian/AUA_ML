from sklearn.linear_model import LinearRegression
import numpy as np

# Sample data (Total bill, Service quality)
X = np.array([[34, 2], [108, 5], [64, 3], [88, 4], [99, 5], [51, 1], [75, 2], [89, 3], [112, 4], [15, 1], [254, 5], [358, 5]])
y = np.array([5, 17, 11, 8, 14, 5, 5, 10, 20, 1, 20, 25])

# Fitting the model
reg = LinearRegression()
reg.fit(X, y)

# Making predictions
new_data = np.array([[100, 3]])  # Total bill = 100, Service quality = 3
predictions = reg.predict(new_data)
print(f"Predicted tip: {predictions[0]:.2f}")
