from sklearn import datasets
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

# Load and preprocess data (same as before)
iris = datasets.load_iris()
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define hyperparameter grid
param_grid = {
    'hidden_layer_sizes': [(10, 10), (50, 50), (100, 100)],  # Explore different layer sizes
    'activation': ['relu', 'tanh', 'logistic'],  # Try different activation functions
    'alpha': [0.0001, 0.001, 0.01],  # Regularization parameter
    'solver': ['lbfgs', 'sgd', 'adam'],
    'learning_rate': ['constant', 'invscaling', 'adaptive'],
}

# Create model and perform grid search
mlp = MLPClassifier(max_iter=400, random_state=42)  # Base model for tuning
grid_search = GridSearchCV(mlp, param_grid, cv=5, scoring='accuracy')  # 5-fold cross-validation
grid_search.fit(X_train, y_train)

# Get best model and evaluate
best_model = grid_search.best_estimator_  # Retrieve best model from grid search
accuracy = best_model.score(X_test, y_test)
print("Best model's accuracy on test set:", accuracy)
print("Best hyperparameters:", grid_search.best_params_)  # View optimal settings
