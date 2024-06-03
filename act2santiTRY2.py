import numpy as np
import os
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from skimage.io import imread

# Función para cargar imágenes desde un directorio y devolverlas como una matriz
def load_images_from_directory(directory_path):
    file_list = os.listdir(directory_path)
    images = []
    for file_name in file_list:
        img = imread(os.path.join(directory_path, file_name), as_gray=True)
        images.append(img.flatten())
    images_matrix = np.array(images)
    return images_matrix, img.shape

# Cargar las imágenes de los directorios
dataset1_path = 'datasets_imgs.zip'
dataset2_path = 'datasets_imgs02.zip'
X1, img_shape = load_images_from_directory(dataset1_path)
X2, _ = load_images_from_directory(dataset2_path)

# 1. Aprender una representación basada en Descomposición de Valores Singulares utilizando las 18 imágenes.
def compute_svd(X, n_components):
    svd = TruncatedSVD(n_components=n_components)
    X_transformed = svd.fit_transform(X)
    return svd, X_transformed

# Visualizar imágenes reconstruidas con distintos valores de d dimensiones
def plot_reconstructed_images(X, svd, img_shape, d_values):
    fig, axes = plt.subplots(len(d_values), 2, figsize=(10, 15))
    for i, d in enumerate(d_values):
        X_approx = svd.inverse_transform(svd.transform(X)[:, :d])
        axes[i, 0].imshow(X[0].reshape(img_shape), cmap='gray')
        axes[i, 0].set_title('Original')
        axes[i, 1].imshow(X_approx[0].reshape(img_shape), cmap='gray')
        axes[i, 1].set_title(f'Reconstrucción con d={d}')
    plt.tight_layout()
    plt.show()

# 2. Visualizar en forma matricial p × p las imágenes reconstruidas
d_values = [5, 10, 50]  # Ejemplo de valores de d
svd, _ = compute_svd(X1, n_components=max(d_values))
plot_reconstructed_images(X1, svd, img_shape, d_values)

# 3. Medir la similaridad entre pares de imágenes en un espacio de baja dimensión d
def compute_similarity(X, d):
    svd, X_transformed = compute_svd(X, n_components=d)
    similarity_matrix = cosine_similarity(X_transformed)
    return similarity_matrix

# Visualizar la matriz de similaridad para distintos valores de d
def plot_similarity_matrices(X, d_values):
    fig, axes = plt.subplots(1, len(d_values), figsize=(15, 5))
    for i, d in enumerate(d_values):
        similarity_matrix = compute_similarity(X, d)
        im = axes[i].imshow(similarity_matrix, cmap='hot', interpolation='nearest')
        axes[i].set_title(f'Similarity matrix d={d}')
        plt.colorbar(im, ax=axes[i])
    plt.tight_layout()
    plt.show()

plot_similarity_matrices(X1, d_values)

# 4. Encontrar el número mínimo de dimensiones d tal que el error no exceda el 10% bajo la norma de Frobenius
def find_optimal_d(X, max_d=100, error_threshold=0.1):
    errors = []
    svd = TruncatedSVD(n_components=max_d)
    X_transformed = svd.fit_transform(X)
    for d in range(1, max_d + 1):
        X_approx = svd.inverse_transform(X_transformed[:, :d])
        error = np.linalg.norm(X - X_approx, 'fro') / np.linalg.norm(X, 'fro')
        errors.append((d, error))
        if error <= error_threshold:
            return d, errors
    return max_d, errors

optimal_d, errors = find_optimal_d(X2)
print(f"Optimal d for dataset 2: {optimal_d}")

# Utilizar la base de d dimensiones obtenida del dataset 2 para reconstruir dataset 1 y calcular el error de reconstrucción
svd, _ = compute_svd(X2, n_components=optimal_d)
X1_approx = svd.inverse_transform(svd.transform(X1))
reconstruction_error = np.linalg.norm(X1 - X1_approx, 'fro') / np.linalg.norm(X1, 'fro')
print(f"Reconstruction error for dataset 1 using base from dataset 2 with d={optimal_d}: {reconstruction_error}")

# Graficar el error vs. número de dimensiones
d_values, errors = zip(*errors)
plt.plot(d_values, errors, marker='o')
plt.axhline(y=0.1, color='r', linestyle='--', label='10% Error Threshold')
plt.xlabel('Number of dimensions (d)')
plt.ylabel('Reconstruction Error')
plt.title('Reconstruction Error vs. Number of Dimensions')
plt.legend()
plt.show()
