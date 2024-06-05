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

def svd_d(X, d):
    U, S, Vt = np.linalg.svd(X, full_matrices=False)

    U_reducido = U[:, :d]
    S_reducido = np.diag(S[:d])
    Vt_reducido = Vt[:d, :]

    X_reducida = U_reducido.dot(S_reducido).dot(Vt_reducido)
    return X_reducida

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
    for j in range(i+1, 6):
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
        axes[i].set_title(f'Matriz de Similaridad d={d_values[i]}', fontsize=11, fontstyle='italic', fontweight='bold')
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
        axes[i].set_title(f'Imagen {i + 1}', fontsize=10)
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


    #mostrar solo las imagenes 2 y 17 cada una con calculada con dimension 5
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle('Reconstrucción de Imágenes 12 y 8 con d=5', fontsize=10)
    axes[0].imshow(X_approx[11].reshape(img_shape), cmap='gray')
    axes[0].set_title('Imagen 12', fontsize=10)
    axes[0].axis('off')
    axes[1].imshow(X_approx[7].reshape(img_shape), cmap='gray')
    axes[1].set_title('Imagen 8', fontsize=10)
    axes[1].axis('off')
    plt.tight_layout(rect=[0, 0, 1, 0.9])  # Ajusta el layout para dejar espacio para el título
    plt.show()

    #mostrar solo las imagenes 2 y 17 cada una con calculada con dimension 5
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle('Reconstrucción de Imágenes 2 y 17 con d=5', fontsize=10)
    axes[0].imshow(X_approx[1].reshape(img_shape), cmap='gray')
    axes[0].set_title('Imagen 2', fontsize=10)
    axes[0].axis('off')
    axes[1].imshow(X_approx[17].reshape(img_shape), cmap='gray')
    axes[1].set_title('Imagen 17', fontsize=10)
    axes[1].axis('off')
    plt.tight_layout(rect=[0, 0, 1, 0.9])  # Ajusta el layout para dejar espacio para el título
    plt.show()

X1, img_shape = load_images_from_directory(dataset1_path)
X2, _ = load_images_from_directory(dataset2_path)



# grafico_errores()
representacion_svd()
# imagenes_d_fijo()
# plot_reconstructed_images(X1, img_shape)
# plot_similarity_matrices(X1, d_values)


#2.4
'''
Optimal d for dataset 2: 8
Reconstruction error for dataset 1 using base from dataset 2 with d=8: 0.7405552594410275
Conclusion: el unico rango que aproxima con un error menor al 10% es el rango de 8 dimensiones, con un error de 0.74%.
'''















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
