# Haz una funcion de factorizacion PA=LU

import numpy as np

def factorizacionLU(A):
    n = len(A)
    L = np.eye(n)
    U = np.zeros((n,n))
    for i in range(n):
        for j in range(i,n):
            sum = 0
            for k in range(i):
                sum += L[i][k]*U[k][j]
            U[i][j] = A[i][j] - sum
        for j in range(i+1,n):
            sum = 0
            for k in range(i):
                sum += L[j][k]*U[k][i]
            L[j][i] = (A[j][i] - sum)/U[i][i]
    return L, U