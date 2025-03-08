# K-Nearest Neighbors (KNN) Classifier

This repository contains an implementation of a K-Nearest Neighbors (KNN) classifier from scratch in Python. The classifier supports three different distance metrics: Euclidean, Manhattan, and Cosine similarity.

## Features

- Supports **Euclidean**, **Manhattan**, and **Cosine** distance metrics.
- Implements a vectorized approach for finding nearest neighbors.
- Includes functionality for training, predicting, and evaluating accuracy.
- Compatible with **NumPy** and **Pandas** for handling datasets.
- Demonstrates functionality using synthetic data generated with **Scikit-learn**.

## Installation

Ensure you have Python installed along with the following dependencies:

```sh
pip install numpy pandas matplotlib scikit-learn
```

## Usage

### Importing and Initializing the Classifier

```python
from knn import KNeighborsClassifier, EUCLIDIAN, MANHATTAN, COSINE
```

### Creating and Training the Model

```python
from sklearn.datasets import make_blobs
import pandas as pd

# Generate a synthetic dataset
X, y = make_blobs(n_samples=200, centers=3, n_features=3, random_state=1)

# Split data into training and testing sets
X_train, y_train = X[:150], y[:150]
X_test, y_test = X[150:], y[150:]

# Convert training data into a Pandas DataFrame
X_train_df = pd.DataFrame({"x": X_train[:, 0], "y": X_train[:, 1], "z": X_train[:, 2]})
y_train_series = pd.Series(y_train)

# Initialize and train the model
knn = KNeighborsClassifier(n_neighbors=7, metric=MANHATTAN)
knn.fit(X_train_df, y_train_series)
```

### Making Predictions

```python
y_pred = knn.predict(X_test)
```

### Evaluating Model Performance

```python
accuracy = knn.score(X_test, y_test)
print(f"Model Accuracy: {accuracy}")
```

### Visualizing the Results

```python
import matplotlib.pyplot as plt

plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, marker='o', label="Training Data")
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, marker='+', label="Predictions")
plt.legend()
plt.show()
```

## Distance Metrics

- **Euclidean (0)**: Standard distance formula.
- **Manhattan (1)**: Sum of absolute differences.
- **Cosine (2)**: Measures cosine similarity.

## Notes

- The classifier expects input data as a **NumPy array** or a **Pandas DataFrame**.
- Ensure labels are provided as a **NumPy array** or a **Pandas Series**.

## License

This project is licensed under the MIT License.

## Author

JohnPaul Vela

