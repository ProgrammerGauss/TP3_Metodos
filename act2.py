# from matplotlib.image import *
# from pylab import *
# #from google.colab import drive
# #drive.mount('/content/drive')
# import matplotlib.pyplot as plt
# import numpy as np
# import os

# A = np.random.rand(5,3)
# U, S, V = np.linalg.svd(A, full_matrices=True)
# Uhat, Shat, Vhat = np.linalg.svd(A, full_matrices=False)

# print(U.shape, S.shape, V.shape)
# print(Uhat.shape, Shat.shape, Vhat.shape)

# A = imread('path')
# print(A.shape)
# plt.imshow(A)

# X = np.mean(A, -1)
# plt.imshow(X, cmap = "gray")

# U, S, VT = np.linalg.svd(X, full_matrives = False)
# S = np.diag(S)
# print(U.shape, S.shape, VT.shape)

# j = 0
# for r in (5, 20 ,100):
#     Xaproxx = U[:,:r]@S[:r,:r]@VT[:r,:]
#     plt.figure(j+1)
#     img = plt.imshow(Xaproxx, cmap = 'gray')
#     plt.title(f'r={r}')



'''
el archivo dataset_imgs.zip se encuentran n imágenes. Cada imagen es una matriz de p × p que
puede representarse como un vector x ∈ Rp∗p. A su vez, es posible armar un matriz de datos apilando los
vectores de cada imagen generando una matriz de n × (p ∗ p). Se desea aprender una representación de baja
dimensión de las imágenes mediante una descomposición en valores singulares

1. Hacer una representacion basada en  Descomposición de Valores Singulares utilizando las n imágenes
'''

#como haria lo que me piden en el punto uno? 
#