import numpy as np
import pandas as pd
from sklearn.decomposition import *
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler


#cargar los datos del csv y y.txt
data = pd.read_csv('dataset03.csv', header=None)
data = data.drop(data.columns[0], axis=1)
data = data.drop(data.index[0], axis=0)

X = data.to_numpy()
X = X.astype(float)

y = np.loadtxt('Datasets/y3.txt')

sigma = 1.0
dimensions = [2]

def similarity(xi, xj, sigma):
    dist_sq = np.linalg.norm(xi - xj) ** 2
    return np.exp(-dist_sq / (2 * sigma ** 2))

# calculo PCA utilizando SVD
def PCA(X, d):
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_mean = np.mean(X, axis=0)
    X_centered = X - X_mean
    covariance_matrix = np.cov(X_centered.T)  # Note the transpose
    U, S, VT = np.linalg.svd(covariance_matrix, full_matrices=False)

    X_reducido = U[:,:d] @ np.diag(S[:d]) @ VT[:d,:]
    return X_reducido

def calculate_similarity_matrix(X, sigma):
    n_samples = X.shape[0]
    similarity_matrix = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(n_samples):
            similarity_matrix[i, j] = similarity(X[i], X[j], sigma)
    return similarity_matrix

# Calcular la matriz de similaridad para los datos originales y los datos reducidos
def act1_1():
    for d in dimensions:

        X_reducido = PCA(X, d)

        similarity_matrix_X = calculate_similarity_matrix(X, sigma)
        similarity_matrix_X_reducido = calculate_similarity_matrix(X_reducido, sigma)

        # Visualizar las matrices de similaridad
        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        sns.heatmap(similarity_matrix_X, cmap='viridis')
        plt.title(f'Similarity Matrix for Original Data (d={d})')

        plt.subplot(1, 2, 2)
        sns.heatmap(similarity_matrix_X_reducido, cmap='viridis')
        plt.title('Similarity Matrix for Reduced Data')

        plt.show()

def act1_3():
    def PCA2(X, d):
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        X_mean = np.mean(X, axis=0)
        X_centered = X - X_mean
        covariance_matrix = np.cov(X_centered.T)  # Note the transpose
        U, S, VT = np.linalg.svd(covariance_matrix, full_matrices=False)

        U_reducido = U[:,:d]
        S_reducido = np.diag(S[:d])
        V_reducido = VT[:d,:]

        return U_reducido, S_reducido, V_reducido
    
    # Resolver el problema de cuadrados mínimos y calcular el error de predicción para cada dimensión
    errors = []
    for d in range(1, X.shape[1] + 1):
        U_reducido, S_reducido, V_reducido = PCA2(X, d)
        X_reducido = X @ U_reducido
        beta_reducido = np.linalg.pinv(X_reducido.T @ X_reducido) @ X_reducido.T @ y
        y_pred = X_reducido @ beta_reducido
        error = np.linalg.norm(y - y_pred)**2
        errors.append(error)

    # Encontrar la dimensión que minimiza el error de predicción
    min_error = min(errors)
    best_d = errors.index(min_error) + 1
    print("La dimensión que minimiza el error de predicción es: ", best_d)

    # Encontrar las muestras con mejor predicción
    U_reducido, S_reducido, V_reducido = PCA2(X, best_d)
    X_reducido = X @ U_reducido
    beta_reducido = np.linalg.pinv(X_reducido.T @ X_reducido) @ X_reducido.T @ y
    y_pred = X_reducido @ beta_reducido
    residuals = np.abs(y - y_pred)
    best_samples = np.argsort(residuals)[:10]  # Asume que queremos las 10 mejores muestras
    print("Las muestras con mejor predicción son: ", best_samples)

    # Crear un gráfico que muestre el error de predicción para cada dimensión después de aplicar PCA
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, X.shape[1] + 1), errors, marker='o')
    plt.xlabel('Dimensiones')
    plt.ylabel('Error de predicción')
    plt.title('Error de predicción para cada dimensión después de aplicar PCA')
    plt.show()

    # Crear un gráfico que muestre los residuos para las mejores muestras
    plt.figure(figsize=(10, 6))
    plt.bar(range(1, 11), residuals[best_samples])
    plt.xlabel('Muestras')
    plt.ylabel('Residuos')
    plt.title('Residuos para las mejores muestras')
    plt.show()

