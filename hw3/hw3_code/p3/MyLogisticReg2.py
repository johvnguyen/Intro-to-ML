import pandas as pd
import numpy as np
import time

class MyLogisticReg2:
  def __init__(self, d):
    # Regression params; adjust dimension for extra column
    self.dim = d+1
    self.w_1 = np.random.uniform(-0.1, 0.1, d+1) # Dummy initial params
    
    #GD params
    self.learning_rate = 1e-8
    
  def fit(self, X, y):
    # initial params suggested by textbook
    self.w_1 = np.random.uniform(-00.1, 0.01, self.dim)
    
    # Use gradient descent to optimize w_1 and w_0 by adding an extra column so we only have to optimize one parameter: w_1
    self.add_col(X)
    
    # Dummy initial values
    old_loss = -20
    loss = -10
    
    n_data, n_features = X.shape
    
    while(abs(old_loss - loss) > 1e-4):
      old_loss = loss
      
      gradient = np.zeros(self.dim)
      z = np.dot(X.values, self.w_1)
      predict = self.sigmoid(z)
      
      gradient = np.dot(X.values.T, (predict - y.values)) / n_features
      
      self.w_1 = self.w_1 - (self.learning_rate)*(gradient)
      
      loss = self.loss(predict, y)
      
    return
    
  def predict(self, X):
    # Returns predictions for input data X
    z = np.dot(X.values, self.w_1)
    prob = self.sigmoid(z)
    
    ret = np.where(prob >= 0.5, 1, 0)
    
    
    return ret
    
  def score(self, X, y):
    self.add_col(X)
    prediction = self.predict(X)
    
    n_data, _ = X.shape
    
    correct = 0
    
    for i in range(n_data):
      if prediction[i] == y.iloc[i]:
        correct += 1
    
    accuracy = correct / n_data
    return accuracy
  
  def sigmoid(self, z):
    return 1.0 / (1 + np.exp(-z))
    
  def add_col(self, X):
    new_col = np.ones((X.shape[0], 1))
    X.insert(0, "adj", new_col, True)
    return
    
  def loss(self, h, y):
    loss_val = ((-y.values * np.log(h)) - ((1 - y.values) * np.log(1 - h)))
    length = len(loss_val)
    
    loss_val = loss_val.sum()
    loss_val = loss_val / length
    
    return loss_val
    
  