import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

def perceptron_train(X, y, learning_rate=0.01, epochs=100):
    # Add a bias term to the input features
    X_bias = np.c_[np.ones((X.shape[0], 1)), X]

    # Initialize weights randomly
    np.random.seed(42)
    weights = np.random.rand(X_bias.shape[1])

    for epoch in range(epochs):
        for i in range(X_bias.shape[0]):
            # Compute the predicted output
            prediction = np.dot(X_bias[i], weights)

            # Update weights based on the perceptron learning rule
            weights += learning_rate * (y[i] - prediction) * X_bias[i]

    return weights


def perceptron_predict(X, weights):
    # Add a bias term to the input features
    X_bias = np.c_[np.ones((X.shape[0], 1)), X]

    # Make predictions
    predictions = np.dot(X_bias, weights)

    # Apply a threshold to get binary predictions (0 or 1)
    return (predictions >= 0).astype(int)


# Load Iris dataset (using only two classes for simplicity)
iris = load_iris()
print(iris.feature_names)
print(np.unique(iris.target))
X = iris.data[:, :3]  # Use only the first three features for illustration
#X = iris.data
y = (iris.target != 0).astype(int)  # Binary labels (1 for class 1, 0 for class 0)

#plotting data
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot points with color-coded classes
scatter = ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap='viridis', edgecolors='k', s=50)

# Set labels for axes
ax.set_xlabel(iris.feature_names[0])
ax.set_ylabel(iris.feature_names[1])
ax.set_zlabel(iris.feature_names[2])

# Add a colorbar
colorbar = fig.colorbar(scatter, ax=ax, ticks=np.unique(y))
colorbar.set_label('Class')

# Show the plot
plt.title('3D Scatter Plot of Iris Dataset')
plt.show()

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the perceptron
learned_weights = perceptron_train(X_train, y_train)

# Make predictions on the test set
y_pred = perceptron_predict(X_test, learned_weights)
#y_pred_train = perceptron_predict(X_train, learned_weights)

# Evaluate the accuracy
accuracy = accuracy_score(y_test, y_pred)
#accuracy_train = accuracy_score(y_train, y_pred_train)
print("Accuracy:", accuracy)

# Display the learned weights
print("Learned Weights:", learned_weights)
