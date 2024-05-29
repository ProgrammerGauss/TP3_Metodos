import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('dataset03.csv')
X = data.values
X = X[:,1:]
X = X[1:,:]


# Definir la función de similaridad no-lineal
def similarity(xi, xj, sigma):
    dist_sq = np.linalg.norm(xi - xj) ** 2
    return np.exp(-dist_sq / (2 * sigma ** 2))

# Calcular la matriz de similaridad en el espacio original

n_samples = X.shape[0]
sigma = 1.0  # Puedes ajustar el valor de sigma según sea necesario
similarity_matrix_original = np.zeros((n_samples, n_samples))

for i in range(n_samples):
    for j in range(n_samples):
        similarity_matrix_original[i, j] = similarity(X[i], X[j], sigma)

# Reducir la dimensionalidad utilizando PCA y calcular la similaridad
dimensions = [2, 6, 10, X.shape[1]]
similarity_matrices_reduced = {}

for d in dimensions:
    
    pca = PCA(n_components=d)
    X_reduced = pca.fit_transform(X)
    similarity_matrix_reduced = np.zeros((d, d))
    
    for i in range(d):
        for j in range(d):
            similarity_matrix_reduced[i, j] = similarity(X_reduced[i], X_reduced[j], sigma)
    
    similarity_matrices_reduced[d] = similarity_matrix_reduced

# Comparar las medidas de similaridad
similarity_matrices_reduced

#graficar las matrices de similaridad
# plt.figure(figsize=(12, 8))

# for i, d in enumerate(dimensions):
#     plt.subplot(2, 2, i + 1)
#     plt.imshow(similarity_matrices_reduced[d], cmap='rainbow', interpolation='nearest')
#     plt.title(f'Similarity Matrix (d={d})')
#     plt.colorbar()



# plt.tight_layout()
# plt.show()


# Reducir la dimensionalidad utilizando PCA y calcular la similaridad
dimensions = [2, 6, 10, X.shape[1]]
similarity_matrices_reduced = {}

for d in dimensions:
    pca = PCA(n_components=d)
    X_reduced = pca.fit_transform(X)
    similarity_matrix_reduced = np.zeros((d, d))
    
    for i in range(d):
        for j in range(d):
            similarity_matrix_reduced[i, j] = similarity(X_reduced[i], X_reduced[j], sigma)
    
    similarity_matrices_reduced[d] = similarity_matrix_reduced

    # Visualizar la matriz de similaridad reducida utilizando un mapa de calor
    plt.figure(figsize=(10, 8))
    sns.heatmap(similarity_matrix_reduced, cmap='viridis')
    plt.title(f'Matriz de Similaridad en el Espacio Reducido (d={d})')
    plt.xlabel('Muestras')
    plt.ylabel('Muestras')
    plt.show()
