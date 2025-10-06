# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
import numpy as np

class LinearRegressionSGD:
    def _init_(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        # Ensure input is a NumPy array
        X = np.array(X)
        y = np.array(y)

        # Get number of samples and features
        n_samples, n_features = X.shape

        # Initialize weights and bias
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        # Training loop
        for _ in range(self.n_iterations):
            for i in range(n_samples):
                xi = X[i]
                yi = y[i]

                # Prediction
                y_pred = np.dot(xi, self.weights) + self.bias

                # Error
                error = y_pred - yi

                # Gradient update
                self.weights -= self.learning_rate * error * xi
                self.bias -= self.learning_rate * error

    def predict(self, X):
        X = np.array(X)
        return np.dot(X, self.weights) + self.bias

# Example usage
if _name_ == "_main_":
    # Sample dataset: 3 features
    X = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
        [10, 11, 12]
    ])
    y = np.array([6, 15, 24, 33])  # Target values

    # Create and train model
    model = LinearRegressionSGD(learning_rate=0.001, n_iterations=1000)
    model.fit(X, y)

    # Make predictions
    predictions = model.predict(X)

    # Output results
    print("Weights:", model.weights)
    print("Bias:", model.bias)
    print("Predictions:", predictions)
```

## Output:
![alt text](<Screenshot 2025-10-07 004153.png>)


## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
