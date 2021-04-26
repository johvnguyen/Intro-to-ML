from sklearn import datasets
from sklearn import svm           # for SVC and LinearSVC
from sklearn import linear_model  # for LogisticRegression
import pandas as pd
import numpy as np

"""
HELPER FUNCTIONS
"""

def getBoston50():
  boston = datasets.load_boston()
  df = pd.DataFrame(np.c_[boston['data'], boston['target']],
                    columns= np.append(boston['feature_names'], ['target']))
  
  median = df['target'].median()
  n = df.shape[0]
  
  for row in range(n):
    result = df.iloc[row]['target']
    
    if (result >= median):
      df.iloc[row]['target'] = 1
    else:
      df.iloc[row]['target'] = 0
      
  return df
  
def getBoston75():
  boston = datasets.load_boston()
  df = pd.DataFrame(np.c_[boston['data'], boston['target']],
                    columns= np.append(boston['feature_names'], ['target']))
  
  quant = df['target'].quantile(0.75)
  n = df.shape[0]
  
  for row in range(n):
    result = df.iloc[row]['target']
    
    if (result >= quant):
      df.iloc[row]['target'] = 1
    else:
      df.iloc[row]['target'] = 0
      
  return df
  
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
  
def my_split(X, y, pi):
  test = []
  
  n = len(X.index)
  all_indices = range(n)
  
  train = list(all_indices)
  select = int(np.floor(n * pi))
  
  for i in range(select):
    choice = np.random.choice(train)
    
    test.append(choice)
    train.remove(choice)
  
  X_train = X.iloc[train]
  y_train = y.iloc[train]
  
  X_test = X.iloc[test]
  y_test = y.iloc[test]
  
  return X_train, X_test, y_train, y_test
  
"""
ASSIGNMENT FUNCTIONS
"""

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
    error = 1 - accuracy;
    
    #Print fold results
    print("Fold", iter, ":", error)
    
    iter += 1
    
    errs.append(error)
    
  #Print validation statistics
  print("Mean:", np.mean(errs))
  print("Standard Deviation:", np.std(errs))
  print("\n\n")
  
  
  errs.append(np.mean(errs))
  errs.append(np.std(errs))
  
  #Return the data, just in case it is needed
  return errs

def my_train_test(method, X, y, pi, k):
  errs = []
  
  for i in range(k):
    #Get training and test data
    X_train, X_test, y_train, y_test = my_split(X, y, pi)
    
    #Train the model
    method.fit(X_train, y_train)
    #Test the model
    accuracy = method.score(X_test, y_test)
    error = 1 - accuracy
    errs.append(error)
    
  errs.append(np.mean(errs))
  errs.append(np.std(errs))
    
  return errs
  
"""
MAIN FUNCTIONS
"""

def p3i():
  #Preprocessing begins here
  
  #Loading the datasets; returns Pandas dataframe
  dfb50 = getBoston50()
  dfb75 = getBoston75()
  
  #Loading Digits dataset and converting to Pandas dataframe
  digits = datasets.load_digits()
  data = np.c_[digits.data, digits.target]
  columns = np.append(range(1,65), ["target"])
  df_digits = pd.DataFrame(data, columns=columns)
  
  #Seperating features X, and target y
  #Boston50 and Boston75 have the same features
  b_all_X = dfb50.iloc[:, 0:13]
  b50_all_y = dfb50.iloc[:, 13]
  b75_all_y = dfb75.iloc[:,13]
  
  digits_all_X = df_digits.iloc[:,0:64]
  digits_all_y = df_digits.iloc[:,64]
  
  # Preprocessing ends here
  
  #Creating method objects
  l_svc = svm.LinearSVC()
  svc = svm.SVC()
  log_reg = linear_model.LogisticRegression()
  
  #Running cross validation on the data and models
  #LinearSVC
  print("Error rates for LinearSVC with Boston50:")
  my_cross_val(l_svc, b_all_X, b50_all_y, 10)
  
  print("Error rates for LinearSVC with Boston75:")
  my_cross_val(l_svc, b_all_X, b75_all_y, 10)
  
  print("Error rates for LinearSVC with Digits:")
  my_cross_val(l_svc, digits_all_X, digits_all_y, 10)
  
  #SVC
  print("Error rates for SVC with Boston50:")
  my_cross_val(svc, b_all_X, b50_all_y, 10)
  
  print("Error rates for SVC with Boston75:")
  my_cross_val(svc, b_all_X, b75_all_y, 10)
  
  print("Error rates for SVC with Digits:")
  my_cross_val(svc, digits_all_X, digits_all_y, 10)
  
  #LogisticRegression
  print("Error rates for LogisticRegression with Boston50:")
  my_cross_val(log_reg, b_all_X, b50_all_y, 10)
  
  print("Error rates for LogisticRegression with Boston75:")
  my_cross_val(log_reg, b_all_X, b75_all_y, 10)
  
  print("Error rates for LogisticRegression with Digits:")
  my_cross_val(log_reg, digits_all_X, digits_all_y, 10)
  
  return
  
  
def p3ii():
  #Preprocessing begins here
  
  #Loading the datasets; returns Pandas dataframe
  dfb50 = getBoston50()
  dfb75 = getBoston75()
  
  #Loading Digits dataset and converting to Pandas dataframe
  digits = datasets.load_digits()
  data = np.c_[digits.data, digits.target]
  columns = np.append(range(1,65), ["target"])
  df_digits = pd.DataFrame(data, columns=columns)
  
  #Seperating features, X, and target y
  #Boston50 and Boston75 have the same features
  b_all_X = dfb50.iloc[:, 0:13]
  b50_all_y = dfb50.iloc[:, 13]
  b75_all_y = dfb75.iloc[:,13]
  
  digits_all_X = df_digits.iloc[:,0:64]
  digits_all_y = df_digits.iloc[:,64]
  
  # Preprocessing ends here
  
  #Creating method objects
  l_svc = svm.LinearSVC()
  svc = svm.SVC()
  log_reg = linear_model.LogisticRegression()
  
  # Formatting
  print("\n")
  
  
  #Begin training methods with data
  #LinearSVC
  print("LinearSVC with Boston50")
  b50_l_svc = my_train_test(l_svc, b_all_X, b50_all_y, pi = 0.75, k = 10)
  for i in range(10):
    print("Fold ", i+1, ":", b50_l_svc[i])
  print("Mean:", b50_l_svc[10], "\nStandard Deviation:", b50_l_svc[11], "\n\n")
  
  print("LinearSVC with Boston75")
  b75_l_svc = my_train_test(l_svc, b_all_X, b75_all_y, pi = 0.75, k = 10)
  for i in range(10):
    print("Fold ", i+1, ":", b75_l_svc[i])
  print("Mean:", b75_l_svc[10], "\nStandard Deviation:", b75_l_svc[11], "\n\n")
  
  print("LinearSVC with Digits")
  digits_l_svc = my_train_test(l_svc, digits_all_X, digits_all_y, pi = 0.75, k = 10)
  for i in range(10):
    print("Fold ", i+1, ":", digits_l_svc[i])
  print("Mean:", digits_l_svc[10], "\nStandard Deviation:", digits_l_svc[11], "\n\n")
  
  #SVC
  print("SVC with Boston50")
  b50_svc = my_train_test(svc, b_all_X, b50_all_y, pi = 0.75, k = 10)
  for i in range(10):
    print("Fold ", i+1, ":", b50_svc[i])
  print("Mean:", b50_svc[10], "\nStandard Deviation:", b50_svc[11], "\n\n")
  
  print("SVC with Boston75")
  b75_svc = my_train_test(svc, b_all_X, b75_all_y, pi = 0.75, k = 10)
  for i in range(10):
    print("Fold ", i+1, ":", b75_svc[i])
  print("Mean:", b75_svc[10], "\nStandard Deviation:", b75_svc[11], "\n\n")
  
  print("SVC with Digits")
  digits_svc = my_train_test(svc, digits_all_X, digits_all_y, pi = 0.75, k = 10)
  for i in range(10):
    print("Fold ", i+1, ":", digits_svc[i])
  print("Mean:", digits_svc[10], "\nStandard Deviation:", digits_svc[11], "\n\n")
  
  #LogisticRegression
  print("LogisticRegression with Boston50")
  b50_log_reg = my_train_test(log_reg, b_all_X, b50_all_y, pi = 0.75, k = 10)
  for i in range(10):
    print("Fold ", i+1, ":", b50_log_reg[i])
  print("Mean:", b50_log_reg[10], "\nStandard Deviation:", b50_log_reg[11], "\n\n")
  
  print("LogisticRegression with Boston75")
  b75_log_reg = my_train_test(log_reg, b_all_X, b75_all_y, pi = 0.75, k = 10)
  for i in range(10):
    print("Fold ", i+1, ":", b75_log_reg[i])
  print("Mean:", b75_log_reg[10], "\nStandard Deviation:", b75_log_reg[11], "\n\n")
  
  print("LogisticRegression with Digits")
  digits_log_reg = my_train_test(log_reg, digits_all_X, digits_all_y, pi = 0.75, k = 10)
  for i in range(10):
    print("Fold ", i+1, ":", digits_log_reg[i])
  print("Mean:", digits_log_reg[10], "\nStandard Deviation:", digits_log_reg[11], "\n\n")
