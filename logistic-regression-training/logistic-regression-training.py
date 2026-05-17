import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    # Write code here
    # list to numpy (if not)
    X = np.array(X)
    y = np.array(y)
    n_samples, n_features = X.shape
    # initialize w and b
    w = np.zeros(n_features)
    b = 0.0
    n = X.shape[0]
    for _ in range(steps):
        linear_input = X @ w + b
        y_hat = _sigmoid(linear_input)

        # gradients
        dw = (1/n_samples) * (X.T @ (y_hat-y))
        db = (1/n_samples) * np.sum(y_hat-y)

        # parameter update
        w -= lr * dw 
        b -= lr * db 

    return w,b 
        