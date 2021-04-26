import numpy as np

A = [[1, 1, 1, 1, 1], 
     [1, 2, 3, 4, 5], 
     [1, 3, 9, 27, 81],
     [1, 4, 16, 64, 256], 
     [1, 5, 25, 125, 625]]

A_t = np.transpose(A)
AA_t = A*A_t
A_tA = A_t*A

print(np.trace(A))
print(np.trace(A_t))
print(np.trace(A_tA))
print(np.trace(AA_t))