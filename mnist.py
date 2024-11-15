from sklearn import datasets
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix

digits = datasets.load_digits()
X, y = digits.data, digits.target

#preprocess the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # Normalize features

#splitting data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# svm model and hyperparameter tuning
param_grid = {
    'C': [0.1, 1, 10],  #c is theegularization parameter
    'gamma': [0.001, 0.01, 0.1],  #gamma is the kernel coefficient
    'kernel': ['rbf']  # this is the RBF kernel
}

grid_search = GridSearchCV(SVC(), param_grid, cv=3, verbose=2, n_jobs=-1)
grid_search.fit(X_train, y_train)

#here we evaluate the best model
best_svm = grid_search.best_estimator_
y_pred = best_svm.predict(X_test)


print("Best Parameters:", grid_search.best_params_)
print("Test Set Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


#confusion matrix
cm = confusion_matrix(y_test, y_pred)
labels = np.unique(y_test)  

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.title("Confusion Matrix for MNIST Classification")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.savefig("confusion_matrix.png")
plt.show()

normalized_cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
plt.figure(figsize=(10, 8))
sns.heatmap(normalized_cm, annot=True, fmt='.2%', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.title("Normalized Confusion Matrix for MNIST Classification")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.savefig("normalized_confusion_matrix.png")
plt.show()