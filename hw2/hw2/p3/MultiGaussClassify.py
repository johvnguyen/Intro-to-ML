import pandas as pd
import numpy as np
import time

# Covariance matrix is full
class MultiGaussClassify1:
  def __init__(self, k):
    # Can't set up initial values for mean and covariance until we know how many features each dataset has
    self.n_classes = k
    self.prior = np.full(k, 1/k)
    
    self.mean_mle = []
    
    self.covariance_mle = []
    
    # Placeholders; will be repalced once we fit the model
    for i in range(k):
      self.covariance_mle.append(np.identity(k))
  
  def fit(self, X, y):
    tot_samples, n_features = X.shape
    self.mean_mle = []
    self.covariance_mle = []
    
    for i in range(self.n_classes):
      self.covariance_mle.append(np.identity(n_features))
    
    
    for i in range(self.n_classes):
      # Get all data of class i
      indices = []
      indices = np.where(y == i)
      class_data = X.iloc[indices]
      
      
      n_data,_ = class_data.shape
      
      # Find the mean and correlation matrix for class i
      self.mean_mle = np.append(self.mean_mle, (1/n_data) * np.sum(class_data, axis=0))
      cov_mat = self.get_correlation(class_data, i)
      
      self.covariance_mle[i] = cov_mat
      
      # Set up prior probabilities
      self.prior[i] = n_data/tot_samples
      
    return
  
  
  def predict(self, X):
    tot_data, _ = X.shape
    prediction = [] # Vector of predictions
    
    # Iterate over all data
    for i in range(tot_data):
      # Get current vector of data
      data = X.iloc[i]
      
      # Array of discriminant function values
      discrim_class_array = []
      
      # Iterate over all classes
      for class_iter in range(self.n_classes):
        # Get discriminant function for each class
        prior = self.prior[class_iter]
        conditional = self.getpdf(data, class_iter)
        
        # Check if probability density function returns 0. Avoids divide by zero error.
        if (conditional == 0):
          discrim_class_array = np.append(discrim_class_array, np.NINF)
        else:
          discrim = np.log(prior) + np.log(conditional)
          discrim_class_array = np.append(discrim_class_array, discrim)
        
      # Choose class with highest discriminant function value
      choice = np.argmax(discrim_class_array)
      prediction = np.append(prediction, choice)
      
    return prediction
    
    
  # Probability density function for class i
  def getpdf(self, data, i):
    n_features = data.shape
    n_features = n_features[0]
    
    # Calculate pseudo-determinant to avoid errors
    temp = 1e-3 * np.identity(n_features)
    temp = self.covariance_mle[i] + temp
    
    # Calculate constant term of probability density function
    cov_term = np.sqrt(np.linalg.det(temp))
    pi_term = (2 * np.pi) ** (n_features * (1/2))
    
    const_term = 1/(pi_term * cov_term)
    
    # Calculate exponential term of pdf
    first_term = np.array([data - self.mean_mle[i]])
    inv_cov_term = np.linalg.pinv(temp)
    last_term = np.transpose(first_term)
    
    exp_term = np.exp((-1/2)* (first_term @ inv_cov_term @ last_term))
    
    return const_term * exp_term
  
  # Get error rate
  def score(self, X, y):
    prediction = self.predict(X)
    
    n_data, _ = X.shape
    
    correct = 0
    
    for i in range(n_data):
      if prediction[i] == y.iloc[i]:
        correct += 1
    
    accuracy = correct/n_data
    return accuracy
    
  def get_correlation(self, X, class_index):
    n_data, n_features = X.shape
    matrix = np.zeros((n_features, n_features))
    
    for i in range(n_data):
      data = X.iloc[i]
      term = np.array([(data - self.mean_mle[class_index]).values])
      
      # We transpose the first term b/c this data is row vectors; the formula uses column vectors
      temp = np.transpose(term) @ term
      
      matrix = matrix + temp
      
    matrix = (1/n_data) * matrix
    
    return matrix
    
# Covariance matrix is diagonal
class MultiGaussClassify2:
  def __init__(self, k):
    self.n_classes = k
    self.prior = np.full(k, 1/k)
    
    self.mean_mle = []
    
    self.covariance_mle = []
    
    for i in range(k):
      self.covariance_mle.append(np.identity(k))
  
  def fit(self, X, y):
    tot_samples, n_features = X.shape
    self.mean_mle = []
    self.covariance_mle = []
    
    for i in range(self.n_classes):
      self.covariance_mle.append(np.identity(n_features))
    
    
    for i in range(self.n_classes):
      # Get all data of class i
      indices = []
      indices = np.where(y == i)
      class_data = X.iloc[indices]
      
      
      n_data,_ = class_data.shape
      
      self.mean_mle = np.append(self.mean_mle, (1/n_data) * np.sum(class_data, axis=0))
      cov_mat = self.get_correlation(class_data, i)
      
      cov_mat = np.diag(np.diag(cov_mat))
      
      self.covariance_mle[i] = cov_mat
      
      # Set up prior probabilities
      self.prior[i] = n_data/tot_samples
      
    return
  
  
  def predict(self, X):
    tot_data, _ = X.shape
    prediction = [] # Vector of predictions
    
    # Iterate over all data
    for i in range(tot_data):
      # Get current vector of data
      data = X.iloc[i]
      
      # Array of discriminant function values
      discrim_class_array = []
      
      # Iterate over all classes
      for class_iter in range(self.n_classes):
        # Get discriminant function for each class
        prior = self.prior[class_iter]
        conditional = self.getpdf(data, class_iter)
        
        if (conditional == 0):
          discrim_class_array = np.append(discrim_class_array, np.NINF)
        else:
          discrim = np.log(prior) + np.log(conditional)
          discrim_class_array = np.append(discrim_class_array, discrim)
        
      # Choose class with highest discriminant function value
      choice = np.argmax(discrim_class_array)
      prediction = np.append(prediction, choice)
      
    return prediction
    
    
  # Probability density function for class i
  def getpdf(self, data, i):
    n_features = data.shape
    n_features = n_features[0]
    
    # Calculate pseudo-determinant to avoid errors
    temp = 1e-3 * np.identity(n_features)
    temp = self.covariance_mle[i] + temp
    
    # Calculate constant term of probability density function
    cov_term = np.sqrt(np.linalg.det(temp))
    pi_term = (2 * np.pi) ** (n_features * (1/2))
    
    const_term = 1/(pi_term * cov_term)
    
    # Calculate exponential term of pdf
    first_term = np.array([data - self.mean_mle[i]])
    inv_cov_term = np.linalg.pinv(temp)
    last_term = np.transpose(first_term)
    
    exp_term = np.exp((-1/2)* (first_term @ inv_cov_term @ last_term))
    
    return const_term * exp_term
  
  # Get error rate
  def score(self, X, y):
    prediction = self.predict(X)
    
    n_data, _ = X.shape
    
    correct = 0
    
    for i in range(n_data):
      if prediction[i] == y.iloc[i]:
        correct += 1
    
    accuracy = correct/n_data
    return accuracy
    
  def get_correlation(self, X, class_index):
    n_data, n_features = X.shape
    matrix = np.zeros((n_features, n_features))
    
    for i in range(n_data):
      data = X.iloc[i]
      term = np.array([(data - self.mean_mle[class_index]).values])
      
      # We transpose the first term b/c this data is row vectors; the formula uses column vectors
      temp = np.transpose(term) @ term
      
      matrix = matrix + temp
      
    matrix = (1/n_data) * matrix
    
    return matrix
  