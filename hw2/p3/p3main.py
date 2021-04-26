from sklearn import datasets
from sklearn import linear_model  # for LogisticRegression
import pandas as pd
import numpy as np
import MultiGaussClassify

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
  

def p3main():
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
  mgc_boston_full = MultiGaussClassify.MultiGaussClassify1(2)
  mgc_boston_diag = MultiGaussClassify.MultiGaussClassify2(2)
  
  mgc_digits_full = MultiGaussClassify.MultiGaussClassify1(10)
  mgc_digits_diag = MultiGaussClassify.MultiGaussClassify2(10)
  
  log_reg = linear_model.LogisticRegression()
  
  #Running cross validation on the data and models
  #MultiGauss Classify with Full Covariance Matrix
  print("Error rates for MultiGaussClassify with full covariance matrix on Boston50:")
  my_cross_val(mgc_boston_full, b_all_X, b50_all_y, 5)
  
  print("Error rates for MultiGaussClassify with full covariance matrix on Boston75:")
  my_cross_val(mgc_boston_full, b_all_X, b75_all_y, 5)
  
  print("Error rates for MultiGaussClassify with full covariance matrix on Digits:")
  my_cross_val(mgc_digits_full, digits_all_X, digits_all_y, 5)
  
  #MultiGauss Classify with Diagonal Covariance Matrix
  print("Error rates for MultiGaussClassify with diagonal covariance matrix on Boston50:")
  my_cross_val(mgc_boston_diag, b_all_X, b50_all_y, 5)
  
  print("Error rates for MultiGaussClassify with diagonal covariance matrix on Boston75:")
  my_cross_val(mgc_boston_diag, b_all_X, b75_all_y, 5)
  
  print("Error rates for MultiGaussClassify with diagonal covariance matrix on Digits:")
  my_cross_val(mgc_digits_diag, digits_all_X, digits_all_y, 5)
  
  #LogisticRegression
  print("Error rates for LogisticRegression with Boston50:")
  my_cross_val(log_reg, b_all_X, b50_all_y, 5)
  
  print("Error rates for LogisticRegression with Boston75:")
  my_cross_val(log_reg, b_all_X, b75_all_y, 5)
  
  print("Error rates for LogisticRegression with Digits:")
  my_cross_val(log_reg, digits_all_X, digits_all_y, 5)
  
p3main()