import numpy as np
import os
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from skimage.io import imread
import pandas as pd

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
    fig, axes = plt.subplots(len(d_values), 2, figsize=(10, 15))
    for i, d in enumerate(d_values):
        svd, X_transformed = compute_svd(X, n_components=d)
        X_approx = svd.inverse_transform(X_transformed)
        axes[i, 0].imshow(X[0].reshape(img_shape), cmap='gray')
        axes[i, 0].set_title('Original')
        axes[i, 1].imshow(X_approx[0].reshape(img_shape), cmap='gray')
        axes[i, 1].set_title(f'Reconstrucción con d={d}')
    plt.tight_layout()
    plt.show()

d_values = [5, 10, 50]
plot_reconstructed_images(X1, img_shape, d_values)

def compute_similarity(X, d):
    svd, X_transformed = compute_svd(X, n_components=d)
    similarity_matrix = cosine_similarity(X_transformed)
    return similarity_matrix

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

def find_optimal_d(X, max_d=100, error_threshold=0.1):
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