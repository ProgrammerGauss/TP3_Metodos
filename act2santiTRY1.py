import numpy as np
import os
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from skimage.io import imread

def load_images_from_directory(directory_path):
    file_list = os.listdir(directory_path)
    images = []
    for file_name in file_list:
        img = imread(os.path.join(directory_path, file_name), as_gray=True)
        images.append(img.flatten())
    images_matrix = np.array(images)
    return images_matrix, img.shape

dataset1_path = 'Datasets/datasets_imgs'
dataset2_path = 'Datasets/datasets_imgs_02'

X1, img_shape = load_images_from_directory(dataset1_path)
X2, _ = load_images_from_directory(dataset2_path)

def compute_svd(X, n_components):
    svd = TruncatedSVD(n_components=n_components)
    X_transformed = svd.fit_transform(X)
    return svd, X_transformed

def plot_reconstructed_images(X, img_shape, d_values):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))  # Cambiamos a 2 filas y 3 columnas
    axes = axes.ravel()  # Aplanamos el array de ejes para poder indexarlo con un solo índice

    for i, d in enumerate(d_values):
        svd, X_transformed = compute_svd(X, n_components=d)
        X_approx = svd.inverse_transform(X_transformed)
        axes[i].imshow(X_approx[0].reshape(img_shape), cmap='gray')  # Mostramos la imagen reconstruida
        axes[i].set_title(f'Reconstrucción con d={d}')

    # Mostramos la imagen original en la última posición
    axes[-1].imshow(X[0].reshape(img_shape), cmap='gray')
    axes[-1].set_title('Original')

    plt.tight_layout()
    plt.show()

d_values = [2, 5, 8, 12, 16]
plot_reconstructed_images(X1, img_shape, d_values)

def similaridad_con_producto_escalar_y_norma(X, d):
    _, X_transformed = compute_svd(X, d)
    X_transformed_normalized = X_transformed / np.linalg.norm(X_transformed, axis=1)[:, None]
    similarity_matrix = X_transformed_normalized @ X_transformed_normalized.T
    return similarity_matrix

# def compute_similarity(X, d):
#     _, X_transformed = compute_svd(X, d)
#     similarity_matrix = cosine_similarity(X_transformed)
#     return similarity_matrix

def plot_similarity_matrices(X, d_values):
    _, axes = plt.subplots(1, len(d_values), figsize=(15, 5))
    for i, d in enumerate(d_values):
        similarity_matrix = similaridad_con_producto_escalar_y_norma(X, d)
        im = axes[i].imshow(similarity_matrix, cmap='hot', interpolation='nearest')
        axes[i].set_title(f'Similarity matrix d={d}')
        plt.colorbar(im, ax=axes[i])
    plt.tight_layout()
    plt.show()

plot_similarity_matrices(X1, d_values)

def find_optimal_d(X, max_d=8, error_threshold=0.1):
    errors = []
    for d in range(1, max_d + 1):
        svd, X_transformed = compute_svd(X, n_components=d)
        X_approx = svd.inverse_transform(X_transformed)
        error = np.linalg.norm(X - X_approx, 'fro') / np.linalg.norm(X, 'fro')
        errors.append((d, error))
        if error <= error_threshold:
            return d, errors
    return max_d, errors

optimal_d, errors = find_optimal_d(X2)
print(f"Optimal d for dataset 2: {optimal_d}")

svd, _ = compute_svd(X2, n_components=optimal_d)
X1_approx = svd.inverse_transform(svd.transform(X1))
reconstruction_error = np.linalg.norm(X1 - X1_approx, 'fro') / np.linalg.norm(X1, 'fro')
print(f"Reconstruction error for dataset 1 using base from dataset 2 with d={optimal_d}: {reconstruction_error}")

d_values, errors = zip(*errors)
plt.plot(d_values, errors, marker='o')
plt.axhline(y=0.1, color='r', linestyle='--', label='10% Error Threshold')
plt.xlabel('Number of dimensions (d)')
plt.ylabel('Reconstruction Error')
plt.title('Reconstruction Error vs. Number of Dimensions')
plt.legend()
plt.show()

#Aprender una representación basada en Descomposición de Valores Singulares utilizando las n imágenes del dataset 1. 
X1_svd = np.linalg.svd(X1, full_matrices=False)
U1, S1, V1 = X1_svd

#graficar los valores singulares de la matriz X1, como un grafico de barras
plt.bar(range(1, len(S1) + 1), S1)
plt.xlabel('Singular Value Index')
plt.ylabel('Singular Value')
plt.title('Singular Values of X1')
plt.show()

d_values = [i for i in range(1, 20)]
print(d_values)



# Conclusiones
'''
2.2:
En cuanto a las conclusiones, al visualizar las imágenes reconstruidas después de la compresión con diferentes 
valores de d dimensiones, puedes observar cómo la calidad de la reconstrucción mejora a medida que aumenta el 
número de dimensiones. Con un valor de d bajo, la imagen reconstruida puede ser apenas reconocible, pero a medida que d aumenta
, la imagen se vuelve cada vez más similar a la original. Esto se debe a que un mayor número de dimensiones permite capturar más
 información de la imagen original. Sin embargo, también hay un compromiso, ya que un mayor número de dimensiones también significa
 un mayor costo computacional.'''

'''
2.3:
Utilizando compresión con distintos valores de d medir la similaridad entre pares de imágenes (con
alguna métrica de similaridad que decida el autor) en un espacio de baja dimensión d. Analizar cómo
la similaridad entre pares de imágenes cambia a medida que se utilizan distintos valores de d. Cuales
imágenes se encuentran cerca entre si? Alguna interpretación al respecto? Ayuda: ver de utilizar una
matriz de similaridad para visualizar todas las similaridades par-a-par juntas.'''
