CSCI 5521 Homework 2; John Nguyen; ID: 5116263; email: nguy2539@umn.edu

This folder has my assignment and the associated code.

hw2.pdf                       - My answers for the homework problems.
p3/MultiGaussClassifty.py     - Code associated with Problem 3; the MultiGaussClassify class
p3/p3main.py                  - Code associated with Problem 3. To use, run p3main.py using python3.

Notes - p3/MultiGaussClassifty
  Based on the instructions, there is no way to determine how many features are in each dataset when I call MultiGaussClassify1.__init__(k) and MultiGaussClassify2.__init__(k), since k is the number of classes. Therefore, I initialize the list of means of each class as an empty list, and the list of covariance matrices as k k-by-k identity matrices, which will be overwritten when MultiGaussClassify1.fit() and MultiGaussClassify2.fit() is called.
  
  After going to Yingxue's office hours, I was instructed to create 2 classes, one that does MGC with full covariance matrices, and one that does MGC with diagnonal covariance matrices. This is to allow both classes to use the my_cross_val function, since the LogisticRegression.fit() method does not a "diag" parameter. Therefore if I did not seperate MGC into 2 seperate classes, 
  The classes are as follows:
    MultiGaussClassify1 - MGC with full covariance matrices
    MultiGaussClassify1 - MGC with diagonal covariance matrices
  I have found that sometimes, the probability density function returns 0. If I try to take np.log(0), I get a divide by zero error. In order to avoid this, I calcualte the probability density function, and then check if it is 0. If it is, I set the discrminant function for that class equal to -infinity, because if the probability density function is 0, then the sample is unlikely to be part of that class.