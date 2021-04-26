import pandas as pd
import numpy as np

class MySVM2:
  def __init__(self, d):
    self.d = d+1
    self.w = np.random.uniform(-0.1, 0.1, d+1)
    
    self.lmbda = 5
    self.l_rate = 1e-6
    
  def fit(self, X, y):
    self.w = np.random.uniform(-0.1, 0.1, self.d)
    data = self.add_col(X)
    target = self.convert_class(y)
    
    count = 0
    
    n_data, n_features = data.shape
    
    old_loss = -20;
    loss = 10;
    while(abs(old_loss - loss) > 1e-5):
      old_loss = loss
      loss = 0
      
      prediction = self.predict(data.values)
      
      gradient = np.zeros(self.d)
      
      for i in range(len(prediction)):
        # incorrect classification
        if (target.iloc[i] * prediction[i] < 1):
          gradient = (data.values[i] * target.iloc[i] / n_data) - (self.lmbda * self.w)
          
          self.w += self.l_rate * gradient
        # correct classification
        else:
          gradient = self.lmbda * self.w
          
          self.w += -1 * self.l_rate * gradient
      
      loss = self.loss(prediction, target)
      count += 1
      
    return
        
    
  def predict(self, X):
    return np.dot(X, self.w)
    
  def add_col(self, X):
    new_col = np.ones((X.shape[0], 1))
    X.insert(0, "adj", new_col, True)
    return X
    
  def loss(self, h, target): # h is predicted values
    loss = 0
    prod = target * h
    
    for i in range(len(target)):
      if prod.values[i] >= 1:
        loss += 0
      else:
        loss += 1 - prod.values[i]
        
    loss = loss/len(target)
    
    loss += (self.lmbda/2.0 * (np.linalg.norm(self.w) ** 2))
    
    return loss
    
  def score(self, X, y):
    target = self.convert_class(y)
    data = self.add_col(X)
    prediction = self.predict(data.values)
    
    n_data, _ = X.shape
    
    correct = 0
    
    for i in range(n_data):
      if (target.iloc[i] == np.sign(prediction[i])):
        correct += 1
        
    accuracy = correct / n_data
    return accuracy
    
  def convert_class(self, y):
    for i in range(len(y)):
      if (y.iloc[i] == 0):
        y.iloc[i] = -1
    return y
    