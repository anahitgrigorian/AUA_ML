from sklearn.linear_model import LogisticRegression
import numpy as np

# Sample data (Exam Score 1, Exam Score 2)
X = np.array([[34, 88], [108, 75], [64, 70], [88, 95], [99, 50], [51, 60], [75, 85]])
y = np.array([1, 0, 1, 1, 0, 0, 1])  # Pass (1) or Fail (0)

# Fit the logistic regression model
clf = LogisticRegression()
clf.fit(X, y)

# Predict pass/fail for a new student
new_student = np.array([[90, 80]])  # Scores
prediction = clf.predict(new_student)
print(f"Predicted class (1 = Pass, 0 = Fail): {prediction[0]}")
