import numpy as np
from sklearn.datasets import make_classification

def sigmoid(z):
  return 1.0/(1.0+np.exp(-z))

def calculate_gradient(theta, X, y):
  m = y.shape[0]
  return (X.T @ (sigmoid(X @ theta) - y)) / m

def gradient_descent(X,y, alpha = 0.1, iters = 100, tol = 1e-7):
  X_b = np.c_[np.ones((X.shape[0],1)),X] 
  theta = np.zeros(X_b.shape[1])
  for i in range(iters):
    grad = calculate_gradient(theta,X_b,y)
    theta-= alpha*grad
    if np.linalg.norm(grad)<tol:
      break
  return theta

def predict_prob(X,theta):
   X_b = np.c_[np.ones((X.shape[0],1)),X] 
   return sigmoid(X_b @ theta)

def predict(X,theta,threshold = 0.5):
  return (predict_prob(X,theta) >= threshold).astype(int)

X, y = make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0, random_state=42)
theta = gradient_descent(X, y)
preds = predict(X, theta)

accuracy = (preds == y).mean()
print("Accuracy:", accuracy)
