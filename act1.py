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

sigma = 25
# dimensions = [2, 6, 10, X.shape[1]]

def similarity(xi, xj, sigma):
    dist_sq = np.linalg.norm(xi - xj) ** 2
    return np.exp(-dist_sq / (2 * sigma ** 2))

def PCA(X, d):
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_mean = np.mean(X, axis=0)
    X_centered = X - X_mean
    covariance_matrix = np.cov(X_centered.T)  # Note the transpose
    U, S, VT = np.linalg.svd(covariance_matrix, full_matrices=False)

    U_reducido = U[:,:d]
    X_reducido = X @ U_reducido

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
    dimensions = [2, 6, 10]
    num_dimensions = len(dimensions) + 1  # +1 para los datos originales

    # Calcular la matriz de similaridad para los datos originales
    similarity_matrix_X = calculate_similarity_matrix(X, sigma)

    # Crear una figura con suficientes subplots para todas las dimensiones
    fig, axes = plt.subplots(1, num_dimensions, figsize=(5 * num_dimensions, 5), dpi=100)

    # Ajustar el espaciado entre los subplots
    plt.subplots_adjust(wspace=0.3, hspace=0.3)

    # Mostrar la matriz de similaridad para los datos originales
    sns.heatmap(similarity_matrix_X, cmap='coolwarm', ax=axes[0])
    axes[0].set_title('Datos Originales')
    axes[0].set_xlabel('Índice de Muestra')
    axes[0].set_ylabel('Índice de Muestra')

    # Para cada dimensión, calcular la matriz de similaridad y mostrarla
    for i, d in enumerate(dimensions):
        X_reducido = PCA(X, d)
        similarity_matrix_X_reducido = calculate_similarity_matrix(X_reducido, sigma)
        sns.heatmap(similarity_matrix_X_reducido, cmap='coolwarm', ax=axes[i + 1])
        axes[i + 1].set_title(f'Datos Reducidos (d={d})')
        axes[i + 1].set_xlabel('Índice de Muestra')
        axes[i + 1].set_ylabel('Índice de Muestra')

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


'''de las p dimensiones originales del dataset, cuales son las mas representativas con respecto a las
dimensiones d obtenidas por SVD? Indicar que dimensiones originales del conjunto p son las mas
importantes y el método utilizado para determinarlas.
1
'''
def act1_2():
    def PCA3(X, d):
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

    # Calcular las dimensiones más importantes
    U_reducido, S_reducido, V_reducido = PCA3(X, 2)
    V = V_reducido.T
    most_important_dimensions = np.argsort(np.abs(V), axis=0)[-2:]
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

act1_1()