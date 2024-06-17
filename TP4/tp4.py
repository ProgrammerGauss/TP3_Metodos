import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as s
import scipy.sparse as sp
import scipy.sparse.linalg as spl
import time

def gradiente_de_funcion_de_costo(A, b, x):
    np.seterr(over='ignore')
    result = np.dot(A.T, np.dot(A, x) - b) * 2
    np.seterr(over='warn')
    return result

def gradiente_de_funcion_de_costo_reg(A, b, x, delta):
    np.seterr(over='ignore')
    result = np.dot(A.T, np.dot(A, x) - b) * 2 + 2 * delta * x
    np.seterr(over='warn')
    return result
def gradiente_descendente(A, b, s, x0, n_iter):
    x = x0
    for i in range(n_iter):
        x = x - s * gradiente_de_funcion_de_costo(A, b, x)
    return x

def gradiente_descendente_reg(A, b, s, x0, n_iter, delta):
    x = x0
    for i in range(n_iter):
        x = x - s * gradiente_de_funcion_de_costo_reg(A, b, x, delta)
    return x

def SVD(A, b):
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    x = np.dot(Vt.T, np.dot(np.diag(1/S), np.dot(U.T, b)))
    return x

def calcular_error(A, b, x):
    return np.linalg.norm(np.dot(A, x) - b)

def main():

    #condiciones iniciales
    n = 5
    d = 100
    A = np.random.rand(n, d)
    b = np.random.rand(n)
    x0 = np.random.rand(d)
    s = 1 / np.max(np.linalg.eigvals(np.dot(A.T, A)))
    n_iter = 1000
    delta = 1e-2 * np.linalg.svd(A)[1][0]

    def x_opt():
        return np.dot(np.linalg.pinv(A), b)

    def variacion_de_step(A, b, x0, n_iter):
        A = A / np.linalg.norm(A)
        b = b / np.linalg.norm(b)
        x0 = x0 / np.linalg.norm(x0)

        steps = [1,1e-1, 1e-2, 1e-3, 1e-4, 1e-5, s]
        errores = []
        for step in steps:
            x = gradiente_descendente(A, b, step, x0, n_iter)
            x_reg = gradiente_descendente_reg(A, b, step, x0, n_iter, delta)
            x_svd = SVD(A, b)
            errores.append([calcular_error(A, b, x), calcular_error(A, b, x_reg), calcular_error(A, b, x_svd)])
        plt.figure()
        plt.title('Error en función del step')
        plt.plot(steps, np.array(errores)[:, 0], label='Gradiente descendente')
        plt.plot(steps, np.array(errores)[:, 1], label='Gradiente descendente regularizado')
        plt.plot(steps, np.array(errores)[:, 2], label='SVD')
        plt.xlabel('step')
        plt.ylabel('Error')
        plt.legend()
        plt.xscale('log')
        plt.yscale('log')
        plt.show()
    
    def variacion_de_delta(A, b, x0, n_iter):
        S = np.linalg.svd(A)[1] # Ya vienen ordenados de mayor a menor
        deltas = []
        for i in range(len(S)):
            delta = 1e-2 * S[i]
            deltas.append(delta)
        errores = []
        for delta in deltas:
            x_reg = gradiente_descendente_reg(A, b, s, x0, n_iter, delta)
            errores.append(calcular_error(A, b, x_reg))

        #hacer que deltas_etiquetas tenga para cada delta, la etiqueta: 10^-2 simbolo de sigma y el numero de la lista deltas
        x = np.arange(len(deltas))  # las ubicaciones de las etiquetas
        # plt.rcParams['text.usetex'] = True
        plt.figure()
        plt.title('Error en función de delta')
        plt.bar(x, errores, tick_label=[f'$\sigma_{i}/100$' for i in range(len(deltas))])
        
        plt.ylabel('Error')
        plt.xlabel('Delta')
        plt.yscale('log')
        plt.xticks(rotation=10)
        
        plt.show()

    def variacion_de_iteraciones(A, b, x0):
        
        def y(x):
            return np.linalg.norm(np.dot(A, x) - b)
        
        iters = [5, 30, 150, 1000, 5000, 15000]
        plt.figure()

        for i in range(len(iters)):
            plt.subplot(2, 3, i+1)
            y1 = (gradiente_descendente(A, b, s, x0, iters[i]))
            y2 = (gradiente_descendente_reg(A, b, s, x0, iters[i], delta))
            y3 = (SVD(A, b))
            y4 = (np.dot(np.linalg.inv(np.dot(A.T, A)), np.dot(A.T, b)))
            plt.title(f'Errores con {iters[i]} iteraciones')
            plt.plot(y1, label='Gradiente descendente')
            plt.plot(y2, label='Gradiente descendente regularizado')
            plt.plot(y3, label='SVD')
            plt.plot(y4, label='Cuadrados mínimos')
            plt.plot(x_opt(), label='x', linestyle='--', color='black')
            plt.xlabel('x')
            plt.ylabel('Errores')
            plt.yscale('log')
        plt.tight_layout()
        plt.show()

    def comparacion_de_soluciones(A, b, x0, n_iter, delta):

        def y(x):
            return np.linalg.norm(np.dot(A, x) - b)

        x = gradiente_descendente(A, b, s, x0, n_iter)
        x_reg = gradiente_descendente_reg(A, b, s, x0, n_iter, delta)
        x_svd = SVD(A, b)
        x_least_squares = np.dot(np.linalg.inv(np.dot(A.T, A)), np.dot(A.T, b))
        print('Error gradiente descendente:', y(x))
        print('Error gradiente descendente regularizado:', y(x_reg))
        print('Error SVD:', y(x_svd))
        print('Error cuadrados mínimos:', y(x_least_squares))

        #graficos
        plt.figure()
        plt.title('Comparación de soluciones')
        plt.plot(x, label='Gradiente descendente')
        plt.plot(x_reg, label='Gradiente descendente regularizado')
        plt.plot(x_svd, label='SVD')
        plt.plot(x_least_squares, label='Cuadrados mínimos')
        plt.plot(x_opt(), label='x', linestyle='--', color='black')
        plt.xlabel('x')
        plt.yscale('log')    
        plt.ylabel('y')
        plt.legend()
        plt.show()

    variacion_de_step(A, b, x0, n_iter)
    variacion_de_delta(A, b, x0, n_iter)
    comparacion_de_soluciones(A, b, x0, n_iter, delta)
    variacion_de_iteraciones(A, b, x0)


if __name__ == '__main__':
    main()