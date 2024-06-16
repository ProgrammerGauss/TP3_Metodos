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
    plt.title('Comparación de soluciones')
    plt.plot(x, label='Gradiente descendente')
    # plt.plot(x_reg, label='Gradiente descendente regularizado')
    plt.plot(x_svd, label='SVD')
    # plt.plot(x_least_squares, label='Cuadrados mínimos')
    plt.xlabel('x')
    plt.yscale('log')    
    plt.ylabel('y')
    plt.legend()
    plt.show()

    #variar delta
    deltas = np.logspace(-5, 0, 6)
    errores = []
    for delta in deltas:
        x_reg = gradiente_descendente_reg(A, b, s, x0, n_iter, delta)
        errores.append(np.linalg.norm(np.dot(A, x_reg) - b))
    plt.figure()
    plt.title('Error en función de delta')
    plt.plot(deltas, errores)
    plt.xlabel('delta')
    plt.ylabel('Error')
    plt.xscale('log')
    plt.yscale('log')
    plt.show()
    

#falta el Hesiano, verlo en OneNote



if __name__ == '__main__':
    main()