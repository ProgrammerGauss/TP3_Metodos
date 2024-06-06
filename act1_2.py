import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import pairwise_kernels

# Cargar el dataset
df = pd.read_csv('dataset03.csv')
X = df.values

# Estandarizar los datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Realizar SVD
U, Sigma, Vt = np.linalg.svd(X_scaled, full_matrices=False)

# Función para proyectar los datos
def project_data(X, V, d):
    return np.dot(X, V[:d, :].T)

# Función para calcular similaridades
def calculate_similarity_matrix(Z, sigma):
    return pairwise_kernels(Z, metric='rbf', gamma=1/(2*sigma**2))

# Valores de d a probar
d_values = [2, 6, 10, X.shape[1]]

# Parámetro sigma para la función de similaridad
sigma = 1.0

# Proyectar y visualizar los datos para diferentes valores de d
for d in d_values:
    Z = project_data(X_scaled, Vt, d)
    similarity_matrix = calculate_similarity_matrix(Z, sigma)
    
    plt.figure(figsize=(8, 6))
    plt.scatter(Z[:, 0], Z[:, 1], cmap='viridis')
    plt.title(f'Proyección en {d} dimensiones')
    plt.xlabel('Componente 1')
    plt.ylabel('Componente 2')
    plt.show()

# Varianza explicada por los primeros d componentes
explained_variance = np.cumsum(Sigma**2) / np.sum(Sigma**2)
print("Varianza explicada acumulada:", explained_variance)

# Identificar las dimensiones originales más representativas
most_important_features = np.abs(Vt[:d, :]).sum(axis=0).argsort()[::-1]
print("Dimensiones originales más importantes:", most_important_features[:d])
