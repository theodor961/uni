import time
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

current_time_seed = int(time.time())
# Load the Iris dataset
iris = load_iris()
X = iris.data[:, :2]  # Use only the first two features for simplicity
y = (iris.target != 0).astype(int)  # Binary labels (1 for class 1, 0 for class 0)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=current_time_seed)

# Create and train a perceptron model
perceptron = Perceptron(max_iter=1000, random_state=current_time_seed, eta0=0.1)
perceptron.fit(X_train, y_train)

# Make predictions on the test set
y_pred = perceptron.predict(X_test)

# Evaluate the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: ",accuracy)

# Display the learned weights and bias
print("Learned Weights:", perceptron.coef_)
print("Learned Bias:", perceptron.intercept_)

# Calculate confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=[False, True])

cm_display.plot()
plt.show()