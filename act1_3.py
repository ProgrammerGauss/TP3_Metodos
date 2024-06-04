import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pandas as pd

# Cargar los datos

#cargar los datos del csv y y.txt
data = pd.read_csv('dataset03.csv', header=None)
data = data.drop(data.columns[0], axis=1)
data = data.drop(data.index[0], axis=0)

X = data.to_numpy()
X = X.astype(float)


y = np.loadtxt('Datasets/y3.txt')

# Realizar una regresión lineal en los datos originales
reg = LinearRegression().fit(X, y)
beta = reg.coef_
print(f'Vector β en el espacio original: {beta}')
print(f'Error de predicción en el espacio original: {mean_squared_error(y, reg.predict(X))}')
print(beta.shape)

# Usar PCA para reducir la dimensión de los datos
min_error = np.inf
best_d = None
best_model = None
best_y_pred = None
errors = []
dimensions = list(range(1, X.shape[1] + 1))
for d in dimensions:
    pca = PCA(n_components=d)
    X_reduced = pca.fit_transform(X)
    reg = LinearRegression().fit(X_reduced, y)
    y_pred = reg.predict(X_reduced)
    error = mean_squared_error(y, y_pred)
    errors.append(error)
    if error < min_error:
        min_error = error
        best_d = d
        best_model = reg
        best_y_pred = y_pred

print(f'La mejor dimensión d que mejora la predicción es: {best_d}')

# Identificar las muestras con la mejor predicción
best_samples = np.argsort(np.abs(best_y_pred - y))[:10]
print(f'Las muestras con la mejor predicción son: {best_samples}')

# Graficar el error de predicción para cada dimensión
plt.figure(figsize=(10, 6))
plt.plot(dimensions, errors, marker='o')
plt.xlabel('Dimensión')
plt.ylabel('Error de Predicción')
plt.title('Error de Predicción para cada Dimensión')
plt.grid(True)
plt.show()