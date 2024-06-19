import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as s
import scipy.sparse as sp
import scipy.sparse.linalg as spl
import time

def gradiente_de_funcion_de_costo(A, b, x):
    result = np.dot(A.T, np.dot(A, x) - b) * 2
    return result

def gradiente_de_funcion_de_costo_reg(A, b, x, delta):
    return 2 * (A.T @ (A @ x - b) + delta * x)

def gradiente_descendente(A, b, s, x0, n_iter):
    X_totales = []
    x = x0
    for _ in range(n_iter):
        x = x - s * gradiente_de_funcion_de_costo(A, b, x)
        X_totales.append(x)
    return X_totales

def gradiente_descendente_reg(A, b, s, x0, delta, n_iter):
    X_totales = []
    x = x0
    for _ in range(n_iter):
        x = x - s * gradiente_de_funcion_de_costo_reg(A, b, x, delta)
        X_totales.append(x)
    return X_totales

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
    delta = 1e-1 * np.linalg.svd(A)[1][0]

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
        deltas = [1e-2, 1e-3, 1e-4, 1e-5, delta]
        errores = []
  
        for delta in deltas:
            x_reg = gradiente_descendente_reg(A, b, s, x0, n_iter, delta)
            errores.append(calcular_error(A, b, x_reg))

        #hacer que deltas_etiquetas tenga para cada delta, la etiqueta: 10^-2 simbolo de sigma y el numero de la lista deltas
        x = np.arange(len(deltas))  # las ubicaciones de las etiquetas
        # plt.rcParams['text.usetex'] = True
        plt.figure()
        plt.title('Error en función de delta')
        plt.bar(x, errores, tick_label=deltas)
        
        plt.ylabel('Error')
        plt.xlabel('Delta')
        plt.yscale('log')
        plt.xticks(rotation=10)
        
        plt.show()

    def variacion_de_iteraciones():
        gradiente_dec = gradiente_descendente(A, b, s, x0, n_iter)
        gradiente_dec_reg = gradiente_descendente_reg(A, b, s, x0, delta, n_iter)
        x_svd = SVD(A, b)

        eje_x = np.arange(n_iter)
        errores_gradiente_dec = np.zeros(n_iter)
        errores_gradiente_dec_reg = np.zeros(n_iter)
        errores_x_svd = np.zeros(n_iter)

        for i in range(n_iter):
            errores_gradiente_dec[i] = calcular_error(A, b, gradiente_dec[i])
            errores_gradiente_dec_reg[i] = calcular_error(A, b, gradiente_dec_reg[i])
            errores_x_svd[i] = calcular_error(A, b, x_svd)

        plt.figure()
        plt.title('Variación de iteraciones')
        plt.plot(eje_x, errores_gradiente_dec, label='Gradiente descendente')
        plt.plot(eje_x, errores_gradiente_dec_reg, label='Gradiente descendente regularizado')
        plt.plot(eje_x, errores_x_svd, label='SVD')
        plt.xlabel('Iteraciones')
        plt.ylabel('Error')
        plt.yscale('log')
        plt.legend()
        plt.show()

        


    def comparacion_de_soluciones(A, b, x0, n_iter):
        x = [x0]
        for i in range(1, n_iter + 1):
            x0 = x[i-1]
            x.append(gradiente_descendente(A, b, s, x0))
    
        x_reg = [x0]
        for i in range(1, n_iter + 1):
            x0 = x_reg[i-1]
            x_reg.append(gradiente_descendente_reg(A, b, s, x0, delta))
    
        x_svd = [SVD(A, b) for _ in range(n_iter + 1)]
    
        eje_x = np.arange(n_iter + 1)
        #graficos
        plt.figure()
        plt.title('Comparación de soluciones')
        plt.plot(eje_x, [np.linalg.norm(xi) for xi in x], label='Gradiente descendente')
        plt.plot(eje_x, [np.linalg.norm(xi) for xi in x_reg], label='Gradiente descendente regularizado')
        plt.plot(eje_x, [np.linalg.norm(xi) for xi in x_svd], label='SVD')
        plt.xlabel('Iteraciones')
        plt.ylabel('Error')
        plt.yscale('log')    
        plt.legend()
        plt.show()
    
    

    # variacion_de_step(A, b, x0, n_iter)
    # variacion_de_delta(A, b, x0, n_iter)
    # comparacion_de_soluciones(A, b, x0, n_iter)
    # variacion_de_iteraciones()



if __name__ == '__main__':
    main()