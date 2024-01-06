import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def initialize_weights(input_size, output_size):
    return 0.01 * np.random.randn(input_size, output_size)

def perceptron_train(X, y_one_hot, learning_rate=0.01, epochs=100):
    np.random.seed(42)
    input_size = X.shape[1]
    output_size = y_one_hot.shape[1]

    # Add a bias term to the input features
    X_bias = np.c_[np.ones((X.shape[0], 1)), X]

    # Initialize weights randomly
    weights = initialize_weights(input_size + 1, output_size)

    # Training loop
    for epoch in range(epochs):
        # Compute the predicted output
        scores = np.dot(X_bias, weights)
        #probabilities = softmax(scores)
        #probabilities = 1/(1 + np.exp(-scores))
        # Normalize each row
        m = np.min(scores)
        if m < 0:
            scores = scores - m
        row_sums = np.sum(scores, axis=1, keepdims=True)

        probabilities = scores / row_sums
        # Compute the cross-entropy loss
        loss = -np.sum(y_one_hot * np.log(probabilities)) / len(X)

        # Compute the gradient of the loss with respect to weights
        gradient = np.dot(X_bias.T, (probabilities - y_one_hot)) / len(X)

        # Update weights
        weights -= learning_rate * gradient

        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {loss:.4f}')

    return weights

def perceptron_predict(X, weights):
    # Add a bias term to the input features
    X_bias = np.c_[np.ones((X.shape[0], 1)), X]

    # Make predictions
    scores = np.dot(X_bias, weights)
    probabilities = softmax(scores)

    # Return the index of the maximum probability as the predicted class
    return np.argmax(probabilities, axis=1)

# Load Iris dataset
iris = load_iris()
X = iris.data[:, :2]  # Using the first two features for simplicity
y = iris.target

# One-hot encode the target labels
y_one_hot = np.eye(np.max(y) + 1)[y]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_one_hot, test_size=0.2, random_state=42)

# Train the perceptron
learned_weights = perceptron_train(X_train, y_train)

# Make predictions on the test set
y_pred = perceptron_predict(X_test, learned_weights)

# Convert one-hot encoded predictions back to class labels
y_pred_labels = np.argmax(y_pred, axis=1)

# Evaluate the accuracy
accuracy = accuracy_score(np.argmax(y_test, axis=1), y_pred_labels)
print(f"Accuracy: ",accuracy)

# Display the learned weights
print("Learned Weights:", learned_weights)
