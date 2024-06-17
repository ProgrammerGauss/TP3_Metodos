import numpy as np
import os
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
from skimage.io import imread

dataset1_path = 'Datasets/datasets_imgs'
dataset2_path = 'Datasets/datasets_imgs_02'

d_values = [2, 5, 8, 12, 15, 19]
d_values1 = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15, 16, 17, 18, 19, 20]

def load_images_from_directory(directory_path):
    file_list = os.listdir(directory_path)
    images = []
    for file_name in file_list:
        img = imread(os.path.join(directory_path, file_name), as_gray=True)
        images.append(img.flatten())
    images_matrix = np.array(images)
    return images_matrix, img.shape

def compute_svd(X, n_components):
    svd = TruncatedSVD(n_components=n_components)
    X_transformed = svd.fit_transform(X)
    return svd, X_transformed

def plot_reconstructed_images(X, img_shape):
    fig, axes = plt.subplots(4, 5, figsize=(10, 7), dpi=100, constrained_layout=True)
    axes = axes.ravel()  # Aplanamos el array de ejes para poder indexarlo con un solo índice

    plt.suptitle('Reconstrucción de Imágenes con Diferentes Dimensiones', fontsize=12)
    for i in range(len(d_values1)):
        svd, X_transformed = compute_svd(X, n_components=d_values1[i])
        X_approx = svd.inverse_transform(X_transformed)
        axes[i].imshow(X_approx[0].reshape(img_shape), cmap='gray')
        axes[i].set_title(f'd={d_values1[i]}', fontsize=10)
        axes[i].axis('off')

    # Eliminar los ejes vacíos si no hay suficientes d_values para llenar todas las columnas
    for j in range(i, 6):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

def similaridad_con_producto_escalar_y_norma(X, d):
    _, X_transformed = compute_svd(X, d)
    X_transformed_normalized = X_transformed / np.linalg.norm(X_transformed, axis=1)[:, None]
    similarity_matrix = X_transformed_normalized @ X_transformed_normalized.T
    return similarity_matrix

def plot_similarity_matrices(X, d_values):
    num_rows = np.ceil(len(d_values) / 3).astype(int)
    fig, axes = plt.subplots(num_rows, 3, figsize=(15, 5 * num_rows), dpi=100, constrained_layout=True)
    axes = axes.ravel()  # Aplanamos el array de ejes para poder indexarlo con un solo índice

    for i in range(len(d_values)):
        similarity_matrix = similaridad_con_producto_escalar_y_norma(X, d_values[i])
        im = axes[i].imshow(similarity_matrix, cmap='coolwarm', interpolation='nearest')
        axes[i].set_title(f'Matriz de Similaridad d={d_values[i]}', fontsize=10)
        axes[i].set_xlabel('Índice de Muestra', fontsize=8)
        axes[i].set_ylabel('Índice de Muestra', fontsize=8)
        fig.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)  # Ajustar la barra de colores al tamaño del gráfico

    # Eliminar los ejes vacíos si no hay suficientes d_values para llenar todas las columnas
    for j in range(i+1, num_rows*3):
        fig.delaxes(axes[j])

    plt.show()



def grafico_errores():
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
    d_values, errors = zip(*errors)
    
    svd, _ = compute_svd(X2, n_components=optimal_d)
    X1_approx = svd.inverse_transform(svd.transform(X1))
    reconstruction_error = np.linalg.norm(X1 - X1_approx, 'fro') / np.linalg.norm(X1, 'fro')
    print(f"Reconstruction error for dataset 1 using base from dataset 2 with d={optimal_d}: {reconstruction_error}")

    plt.plot(d_values, errors, marker='o')
    plt.axhline(y=0.1, color='r', linestyle='--', label='10% Error Threshold')
    plt.xlabel('Number of dimensions (d)')
    plt.ylabel('Reconstruction Error')
    plt.title('Reconstruction Error vs. Number of Dimensions')
    plt.gca().set_yticklabels(['{:.0f}%'.format(x*100) for x in plt.gca().get_yticks()])
    y_optimo = errors[6]
    plt.axhline(y=y_optimo, color='g', linestyle='--', label=f'Closest Optimal Error at {y_optimo:.2%}')
    plt.legend()
    plt.show()

def representacion_svd():
    X1_svd = np.linalg.svd(X1, full_matrices=False)
    U1, S1, V1 = X1_svd

    plt.bar(range(1, len(S1) + 1), S1)
    #labels en español
    plt.xlabel('Número de dimensión')
    plt.ylabel('Valor singular')
    plt.title('Valores singulares de X1')
    plt.show()

def imagenes_d_fijo():
    #quiero mostrar como quedan cada una de las imagenes (las 20) pero usando un d fijo
    d = 5
    svd, X_transformed = compute_svd(X1, n_components=d)
    X_approx = svd.inverse_transform(X_transformed)
    fig, axes = plt.subplots(4, 5, figsize=(15, 10))
    axes = axes.ravel()
    
    fig.suptitle(f'Reconstrucción con d={d}', fontsize=10)
    for i in range(19):
        axes[i].imshow(X_approx[i].reshape(img_shape), cmap='gray')
        axes[i].set_title(f'Imagen {i}', fontsize=10)
        axes[i].axis('off')
    
    fig.delaxes(axes[19])
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Ajusta el layout para dejar espacio para el título
    plt.show()

    #mostrar la matriz de similaridad para d=5
    similarity_matrix = similaridad_con_producto_escalar_y_norma(X1, d)
    plt.figure(figsize=(6, 6), dpi=100)
    plt.imshow(similarity_matrix, cmap='coolwarm', interpolation='nearest')
    plt.title(f'Matriz de Similaridad d={d}', fontsize=10)
    plt.xlabel('Índice de Muestra', fontsize=8)
    plt.ylabel('Índice de Muestra', fontsize=8)
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.show()


    #mostrar solo las imagenes 0 y 17 cada una con calculada con dimension 5
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle('Reconstrucción de Imágenes 0 y 17 con d=5', fontsize=30)
    axes[0].imshow(X_approx[0].reshape(img_shape), cmap='gray')
    axes[0].set_title('Imagen 0', fontsize=30)
    axes[0].axis('off')
    axes[1].imshow(X_approx[17].reshape(img_shape), cmap='gray')
    axes[1].set_title('Imagen 17', fontsize=30)
    axes[1].axis('off')
    plt.tight_layout(rect=[0, 0, 1, 0.9])  # Ajusta el layout para dejar espacio para el título
    plt.show()

    #mostrar solo las imagenes 3 y 15 cada una con calculada con dimension 5
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle('Reconstrucción de Imágenes 3 y 15 con d=5', fontsize=30)
    axes[0].imshow(X_approx[3].reshape(img_shape), cmap='gray')
    axes[0].set_title('Imagen 3', fontsize=30)
    axes[0].axis('off')
    axes[1].imshow(X_approx[15].reshape(img_shape), cmap='gray')
    axes[1].set_title('Imagen 15', fontsize=30)
    axes[1].axis('off')
    plt.tight_layout(rect=[0, 0, 1, 0.9])  # Ajusta el layout para dejar espacio para el título
    plt.show()
    
    #mostrar solo las imagenes 0 y 16 cada una con calculada con dimension 5
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle('Reconstrucción de Imágenes 0 y 16 con d=5', fontsize=30)
    axes[0].imshow(X_approx[0].reshape(img_shape), cmap='gray')
    axes[0].set_title('Imagen 0', fontsize=30)
    axes[0].axis('off')
    axes[1].imshow(X_approx[16].reshape(img_shape), cmap='gray')
    axes[1].set_title('Imagen 16', fontsize=30)
    axes[1].axis('off')
    plt.tight_layout(rect=[0, 0, 1, 0.9])  # Ajusta el layout para dejar espacio para el título
    plt.show()

X1, img_shape = load_images_from_directory(dataset1_path)
X2, _ = load_images_from_directory(dataset2_path)


grafico_errores()
representacion_svd()
imagenes_d_fijo()
plot_reconstructed_images(X1, img_shape)
plot_similarity_matrices(X1, d_values)