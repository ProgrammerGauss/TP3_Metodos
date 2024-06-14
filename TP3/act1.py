import numpy as np
import pandas as pd
from sklearn.decomposition import *
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler

#cargar los datos del csv y y.txt
data = pd.read_csv('dataset03.csv', header=None)
data = data.drop(data.columns[0], axis=1)
data = data.drop(data.index[0], axis=0)

X = data.to_numpy()
X = X.astype(float)

y = np.loadtxt('Datasets/y3.txt')

sigma = 3

def similarity(xi, xj, sigma):
    dist_sq = np.linalg.norm(xi - xj) ** 2
    return np.exp(-dist_sq / (2 * sigma ** 2))

def PCA(X, d):
    X_centered = X - np.mean(X, axis=0)
    U, S, VT = np.linalg.svd(X_centered, full_matrices=False)

    for i in range(len(S)):
        if S[i] < 1e-10:
            if i > d:
                S = S[:i]
                break
            else:
                S[i] = 0
    
    S = np.diag(S)
    return U[:,:d] @ S[:d,:d] 

def calculate_similarity_matrix(X, sigma):
    pairwise_dists = squareform(pdist(X, 'euclidean'))
    similarity_matrix = np.exp(-pairwise_dists ** 2 / (2 * sigma ** 2))
    return similarity_matrix

def act1_1():
    dimensions = [2, 6, 10, X.shape[1]]

    # Calcular la matriz de similaridad para los datos originales
    similarity_matrix_X = calculate_similarity_matrix(X, sigma )

    # Crear una figura para los datos originales
    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.matshow(similarity_matrix_X, cmap='viridis')
    fig.colorbar(cax)
    ax.set_title('Datos Originales')
    ax.set_xlabel('Índice de Muestra')
    ax.set_ylabel('Índice de Muestra')
    plt.show()
    
    # Para cada dimensión, calcular la matriz de similaridad y mostrarla en una nueva figura
    for d in dimensions:
        X_reducido = PCA(X, d)
        similarity_matrix_X_reducido = calculate_similarity_matrix(X_reducido, sigma)
    
        fig, ax = plt.subplots(figsize=(6, 6))
        cax = ax.matshow(similarity_matrix_X_reducido, cmap='viridis')
        fig.colorbar(cax)
        ax.set_title(f'Datos Reducidos (d={d})')
        ax.set_xlabel('Índice de Muestra')
        ax.set_ylabel('Índice de Muestra')
        plt.show()

def act1_3():
    def pseudo_inverse(X, d):
        X_centered = X - np.mean(X, axis=0)
        U, S, VT = np.linalg.svd(X_centered, full_matrices=False)
        S_d = np.zeros_like(S)
        S_d[:d] = 1/ S[:d]
        A_pseudo = VT.T @ np.diag(S_d) @ U.T
        return A_pseudo
    
    def beta(X, y, d):
        A_pseudo = pseudo_inverse(X, d)
        beta = A_pseudo @ y
        return beta
    
    def pca(X, d):
        X_centered = X - np.mean(X, axis=0)
        U, S, VT = np.linalg.svd(X_centered, full_matrices=False)
        X_reduced = np.dot(X_centered, VT[:d].T)
        return X_reduced
    
    # Resolver el problema de cuadrados mínimos y calcular el error de predicción para cada dimensión
    errors = []
    dimensiones = [i for i in range(1, X.shape[1] + 1)]

    for d in dimensiones:
        beta_d = beta(pca(X, d), y, d)
        prediccion = np.dot(pca(X, d), beta_d)
        error = np.linalg.norm(prediccion - y) / np.linalg.norm(y)
        errors.append((d, error))

    # Encontrar la dimensión que minimiza el error de predicción
    min_error = min(errors, key=lambda x: x[1])
    best_d = min_error[0]
    print("La dimensión que minimiza el error de predicción es: ", best_d)

    # Encontrar las muestras con mejor predicción
    X_reducido = pca(X, best_d)
    beta_reducido = beta(X_reducido, y, best_d)
    y_pred = np.dot(X_reducido, beta_reducido)
    residuals = np.abs(y - y_pred)
    best_samples = np.argsort(residuals)[:10]  # Asume que queremos las 10 mejores muestras
    print("Las muestras con mejor predicción son: ", best_samples)

    # Crear un gráfico que muestre el error de predicción para cada dimensión después de aplicar PCA
    plt.figure(figsize=(10, 6))
    plt.plot([d for d, _ in errors], [e for _, e in errors], marker='o')
    plt.xlabel('Dimensiones')
    plt.ylabel('Error de predicción')
    plt.title('Error de predicción para cada dimensión después de aplicar PCA')
    plt.show()

    # Crear un gráfico que muestre los residuos para las mejores muestras
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(beta_d)), beta_d.flatten(), marker='o', linestyle='-')
    plt.xlabel('Dimensiones')
    plt.ylabel('Pesos (β)')
    plt.title('Pesos asignados a cada dimensión original')
    ticks_location = np.linspace(0, len(beta_d) - 1, 15, dtype=int)  
    plt.xticks(ticks_location, ticks_location + 1, ha='right')
    plt.show()

def act1_2():
    def PCA3(X, d):

        # return U_reducido, S_reducido, V_reducido
        X_centered = X - np.mean(X, axis=0)
        U, S, VT = np.linalg.svd(X_centered, full_matrices=False)

        for i in range(len(S)):
            if S[i] < 1e-10:
                if i > d:
                    S = S[:i]
                    break
                else:
                    S[i] = 0
        
        S = np.diag(S)
        U_reducido = U[:,:d]
        S_reducido = S[:d,:d]
        V_reducido = VT[:d,:]

        return U_reducido, S_reducido, V_reducido
    
    # Calcular las dimensiones más importantes
    U_reducido, S_reducido, V_reducido = PCA3(X, 2)
    V = V_reducido.T
    most_important_dimensions = np.argsort(np.abs(V), axis=0)[-3:]
    print("Las dimensiones más importantes son: ", most_important_dimensions)

    # Crear un gráfico que muestre la importancia de cada dimensión original
    plt.figure(figsize=(10, 6))
    plt.bar(range(1, X.shape[1] + 1), np.abs(V[:,0]), label='Dimensión 1')
    plt.bar(range(1, X.shape[1] + 1), np.abs(V[:,1]), label='Dimensión 2', alpha=0.5)
    plt.xlabel('Dimensiones originales')
    plt.ylabel('Importancia')
    plt.title('Importancia de cada dimensión original')
    plt.legend()
    plt.show()


def grafico_clusters(X, i):
    X = PCA(X, i)
    Z_centerded = X - np.mean(X, axis=0)
    plt.scatter(Z_centerded[:,0], Z_centerded[:,1], c = np.arange(0, Z_centerded.shape[0]), cmap='coolwarm')
    plt.xlabel('Z1')
    plt.ylabel('Z2')
    plt.title('Clusters')
    plt.colorbar()
    plt.show()

act1_1()
act1_2()
act1_3()
grafico_clusters(X, 2)