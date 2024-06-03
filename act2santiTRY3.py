import zipfile
import numpy as np
from PIL import Image
import io

def read_images_from_zip(zip_path):
    # Abrir el archivo ZIP
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # Listar los archivos en el ZIP
        file_list = zip_ref.namelist()
        
        # Inicializar una lista para almacenar las imágenes
        images = []
        
        # Leer cada archivo en el ZIP
        for file_name in file_list:
            # Abrir el archivo de imagen
            with zip_ref.open(file_name) as file:
                img = Image.open(file)
                # Convertir la imagen en un array de numpy
                img_array = np.array(img)
                # Asegurarse de que la imagen esté en escala de grises
                if len(img_array.shape) > 2:
                    img_array = img_array[:,:,0]
                # Aplanar la imagen en un vector de p*p dimensiones
                img_vector = img_array.flatten()
                # Añadir el vector de la imagen a la lista de imágenes
                images.append(img_vector)
        
        # Convertir la lista de imágenes en una matriz de n x (p*p)
        image_matrix = np.array(images)
        
        return image_matrix

zip_path = 'datasets_imgs.zip'
image_matrix = read_images_from_zip(zip_path)
print(image_matrix.shape)  # Debe imprimir (n, 784) si hay n imágenes en el ZIP
