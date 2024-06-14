'''
1 Cuadrados mínimos mediante descenso por gradiente
El objetivo de este trabajo es aplicar el algoritmo de gradiente descendente al problema de encontrar la
solución del sistema
Ax = b, (1)
donde A ∈ R
n×d
, x ∈ R
d y b ∈ R
n
. Para hacer esto primero definimos la función de costo
F(x) = (Ax − b)
T
(Ax − b). (2)
El algoritmo de gradiente descendente busca encontrar x
∗
, la solución que minimiza F, mediante el proceso
iterativo
xk+1 = xk − s∇F(xk), (3)
donde s es el paso utilizado.
Cuando el problema tiene más incógnitas que ecuaciones se suelen aplicar regularizaciones que inducen
alguna propiedad deseada en la solución obtenida. Una variante típica es agregarle un término que dependa
de la norma-2 del vector al cuadrado a la función de costo, lo que resulta en
F2(x) = F(x) + δ2∥x∥
2
2
, (4)
donde δ2 es un nuevo parámetro a elegir. Esto se conoce como regularización L2. Algunas definiciones antes
de pasar a las consignas: σ son los valores principales de A, λ son los autovalores de H, donde H es el
Hessiano de F (no de F2).
Objetivos:
• Tomando n = 5 y d = 100 genere matrices A y vectores b aleatorios y resuelva el problema minimizando
F y F2. Tome δ2 = 10−2σmax cuando trabaje con F2. En todos los casos utilice s = 1/λmax,
una condición inicial aleatoria y realice 1000 iteraciones. Estudiar como evoluciona la solución del
gradiente descendiente iterativamente. Compare con la solución obtenida mediante SVD. Analice los
resultados. ¿Por qué se elige este valor de s? ¿Qué sucede si se varían los valores de δ2?
'''

import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as s
import scipy.sparse as sp
import scipy.sparse.linalg as spl
import time

def gradiente_descendente(A, b, s, x0, n_iter):
    x = x0
    for i in range(n_iter):
        x = x - s * np.dot(A.T, np.dot(A, x) - b)
    return x

def gradiente_descendente_reg(A, b, s, x0, n_iter, delta):
    x = x0
    for i in range(n_iter):
        x = x - s * (np.dot(A.T, np.dot(A, x) - b) + 2 * delta * x)
    return x

def SVD(A, b):
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    x = np.dot(Vt.T, np.dot(np.diag(1/S), np.dot(U.T, b)))
    return x


def main():
    n = 5
    d = 100
    A = np.random.rand(n, d)
    b = np.random.rand(n)
    x0 = np.random.rand(d)
    s = 1 / np.max(np.linalg.eigvals(np.dot(A.T, A)))
    n_iter = 1000
    delta = 1e-2 * np.max(np.linalg.eigvals(np.dot(A.T, A)))
    x = gradiente_descendente(A, b, s, x0, n_iter)
    x_reg = gradiente_descendente_reg(A, b, s, x0, n_iter, delta)
    x_svd = SVD(A, b)
    print('Error gradiente descendente:', np.linalg.norm(np.dot(A, x) - b))
    print('Error gradiente descendente regularizado:', np.linalg.norm(np.dot(A, x_reg) - b))
    print('Error SVD:', np.linalg.norm(np.dot(A, x_svd) - b))
    x_least_squares = np.dot(np.linalg.inv(np.dot(A.T, A)), np.dot(A.T, b))
    print('Error cuadrados mínimos:', np.linalg.norm(np.dot(A, x_least_squares) - b))

    #graficos
    plt.figure()
    plt.plot(x, label='Gradiente descendente')
    plt.plot(x_reg, label='Gradiente descendente regularizado')
    plt.plot(x_svd, label='SVD')
    # plt.plot(x_least_squares, label='Cuadrados mínimos')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()