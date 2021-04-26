from sklearn import datasets
from sklearn import svm           # for SVC and LinearSVC
from sklearn import linear_model  # for LogisticRegression
import pandas as pd
import numpy as np

def my_cross_val(method, X, y, k):
  #Initialize objects
  errs = []
  iter = 1
  
  kf = kfold(X, k)
  
  for train_index, test_index in kf:
    #Get training data
    train_X = X.iloc[train_index]
    train_y = y.iloc[train_index]
    
    #Get testing data
    test_X = X.iloc[test_index]
    test_y = y.iloc[test_index]
    
    #Train the model
    method.fit(train_X, train_y)
    
    #Test the model
    accuracy = method.score(test_X, test_y)
    error = 1 - accuracy
    
    #Print fold results
    print("Fold", iter, ":", error)
    
    iter += 1
    
    errs.append(error)
    
  #Print validation statistics
  print("Mean:", np.mean(errs))
  print("Standard Deviation:", np.std(errs))
  
  
  errs.append(np.mean(errs))
  errs.append(np.std(errs))
  
  #Return the data, just in case it is needed
  return errs

def kfold(X, k):
  #Create array to return
  ret = []
  
  #Get range of indices of X
  n = len(X.index)
  all_indices = range(n)
  
  #Partition indices of X into k approximately evenly sized chunks
  partitions = np.array_split(all_indices, k)
  
  #Create the train/test split for each fold
  for i in range(k):
    test = partitions[i]
    train = np.setdiff1d(all_indices, test)
    
    ret.append([train, test])
  
  return ret

def rand_proj(X, d):
  #Get dimensions of X
  m, n = X.shape
  
  #Constructing G
  G = np.zeros((n, d))
  for i in range(n):
    for j in range(d):
      G[i,j] = np.random.normal()
      
  #Return the matrix product
  return X.values @ G

def quad_proj(X):
  #Get data to iterate through X
  m, n = X.shape
  X2 = X.values
  
  #Begin making X2 array
  temp = np.multiply(X.values, X.values)
  X2 = np.concatenate((X2, temp), axis = 1)
  
  #Transpose features so I can manipulate columns
  data_transpose = X.values.transpose()
  
  #Iterate over columns
  for j in range(n):
    #Iterate over all columns with greater index than j
    for j2 in range(j+1, n):
      #compute component wise product of columns j and j2
      prod = np.array([np.multiply(data_transpose[j], data_transpose[j2])])
      
      #Transpose product so we have a column again
      temp = prod.transpose()
      
      #Append column to X2
      X2 = np.concatenate((X2, temp), axis = 1)
  
  #Verify X2 has desired size
  assert(X2.shape == (1797, 2144))
  
  return X2
  
  
def p4main():
  #Load the digits dataset and put in dataframe
  digits = datasets.load_digits()
  data = np.c_[digits.data, digits.target]
  columns = np.append(range(1,65), ["target"])
  df_digits = pd.DataFrame(data, columns=columns)
  
  #Seperate features from targets
  digits_all_X = df_digits.iloc[:,0:64]
  digits_all_y = df_digits.iloc[:,64]
  
  #Making X1
  X1 = rand_proj(digits_all_X, 32)
  m, n = X1.shape
  df_X1 = pd.DataFrame(data = X1, index = np.arange(1, m+1), columns = np.arange(1, n+1))
  
  #Making X2
  X2 = quad_proj(digits_all_X)
  m, n = X2.shape
  df_X2 = pd.DataFrame(data = X2, index = np.arange(1, m+1), columns = np.arange(1, n+1))
  
  #Creating method objects
  l_svc = svm.LinearSVC()
  svc = svm.SVC()
  log_reg = linear_model.LogisticRegression()
  
  #LinearSVC
  print("Error rates for LinearSVC with X1:")
  my_cross_val(l_svc, df_X1, digits_all_y, 10)
  
  print("Error rates for LinearSVC with X2:")
  my_cross_val(l_svc, df_X2, digits_all_y, 10)
  print("\n\n")
  
  #SVC
  print("Error rates for SVC with X1:")
  my_cross_val(svc, df_X1, digits_all_y, 10)
  
  print("Error rates for SVC with X2:")
  my_cross_val(svc, df_X2, digits_all_y, 10)
  print("\n\n")
  
  #LogisticRegression
  print("Error rates for LogisticRegression with X1:")
  my_cross_val(log_reg, df_X1, digits_all_y, 10)
  
  print("Error rates for LogisticRegression with X2:")
  my_cross_val(log_reg, df_X2, digits_all_y, 10)
  