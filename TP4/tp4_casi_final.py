import numpy as np
import matplotlib.pyplot as plt

# Definiciones iniciales
n = 5
d = 100
A = np.random.rand(n, d)
b = np.random.rand(n)
x_init = np.random.rand(d)

H_F = 2 * A.T @ A

sigma_max = np.linalg.svd(A, compute_uv=False)[0]
delta2 = 1e-2 * sigma_max
lambda_max = np.linalg.eigvals(H_F).real.max()
s = 1 / lambda_max
iterations = 5000


# Función de costo F(x)
def F(x):
    return np.dot((A @ x - b).T, A @ x - b)

U, S, Vt = np.linalg.svd(A, full_matrices=False)
x_svd = Vt.T @ np.linalg.inv(np.diag(S)) @ U.T @ b
costo_svd = F(x_svd)
norm_svd = np.linalg.norm(x_svd)

# Función de costo F2(x)
def F2(x, delta):
    return F(x) + delta * np.dot(x, x)

# Gradiente de F(x)
def grad_F(x):
    return 2 * A.T @ (A @ x - b)

# Gradiente de F2(x)
def grad_F2(x, delta):
    return grad_F(x) + 2 * delta * x

# Algoritmo de gradiente descendente para F(x)
def gradient_descent_F(A, b, s, x_init, iterations):
    x = x_init
    history_F = []
    history_norm_x = []
    history_residual = []

    for _ in range(iterations):
        x = x - s * grad_F(x)
        history_F.append(F(x))
        history_norm_x.append(np.linalg.norm(x) ** 2)
        history_residual.append(np.linalg.norm(A @ x - b) ** 2)
    
    return x, history_F, history_norm_x, history_residual

# Algoritmo de gradiente descendente para F2(x)
def gradient_descent_F2(A, b, s, x_init, iterations, delta):
    x = x_init
    history_F2 = []
    history_norm_x = []
    history_residual = []

    for _ in range(iterations):
        x = x - s * grad_F2(x, delta)
        history_F2.append(F2(x, delta))
        history_norm_x.append(np.linalg.norm(x) ** 2)
        history_residual.append(np.linalg.norm(A @ x - b) ** 2)
    
    return x, history_F2, history_norm_x, history_residual

def grafico_costo_convergencia():
    x = x_init.copy()
    history_F = []

    x_reg = x_init.copy()
    history_F_reg = []

    for _ in range(iterations):
        grad = grad_F(x)
        if np.any(np.isnan(grad)) or np.any(np.isinf(grad)):
            break
        x = x - s * grad
        history_F.append(F(x))

        grad_reg = grad_F2(x_reg, delta2)
        if np.any(np.isnan(grad)) or np.any(np.isinf(grad)):
            break
        x_reg = x_reg - s * grad_reg
        history_F_reg.append(F(x_reg))

    plt.figure(figsize=(10, 6))
    plt.plot(range(iterations), history_F, label='Decenso por gradiente')
    plt.plot(range(iterations),history_F_reg, label='Decenso por gradiente regularizado')
    plt.axhline(y=costo_svd, color='r', linestyle='--', label='solucion SVD')
    plt.yscale('log')
    plt.xlabel('Iteraciones', fontsize=18)
    plt.ylabel('F(x)', fontsize=18)
    plt.title('F(x) vs Iteraciones', fontsize=20)
    plt.legend(fontsize=18)
    plt.grid()
    plt.show()

def grafico_norma_convergencia():
    x = x_init.copy()
    history_norm_F = []

    x_reg = x_init.copy()
    history_norm_F_reg = []

    for _ in range(iterations):
        grad = grad_F(x)
        if np.any(np.isnan(grad)) or np.any(np.isinf(grad)):
            break
        x = x - s * grad
        history_norm_F.append(np.linalg.norm(x))

        grad_reg = grad_F2(x_reg, delta2)
        if np.any(np.isnan(grad)) or np.any(np.isinf(grad)):
            break
        x_reg = x_reg - s * grad_reg
        history_norm_F_reg.append(np.linalg.norm(x_reg))

    plt.figure(figsize=(10, 6))
    plt.plot(range(iterations), history_norm_F, label='Decenso por gradiente')
    plt.plot(range(iterations), history_norm_F_reg, label='Decenso por gradiente regularizado')
    plt.axhline(y=norm_svd, color='r', linestyle='--', label='solucion SVD')
    plt.xlabel('Iteraciones', fontsize=18)
    plt.ylabel('Norma de x', fontsize=18)
    plt.title('$||x||$ vs Iteraciones', fontsize=20)
    plt.legend(fontsize=18)
    plt.grid()
    plt.show()

def grafico_norma_x():
    x = x_init.copy()
    history_x = [np.linalg.norm(x)]

    x_reg = x_init.copy()
    history_x_reg = [np.linalg.norm(x_reg)]

    for _ in range(iterations):
        grad = grad_F(x)
        if np.any(np.isnan(grad)) or np.any(np.isinf(grad)):
            break
        x = x - s * grad
        history_x.append(np.linalg.norm(x))

        grad_reg = grad_F2(x_reg, delta2)
        if np.any(np.isnan(grad_reg)) or np.any(np.isinf(grad_reg)):
            break
        x_reg = x_reg - s * grad_reg
        history_x_reg.append(np.linalg.norm(x_reg))

    plt.figure(figsize=(10, 6))
    plt.plot(history_x, label='Decenso por gradiente')
    plt.plot(history_x_reg, label='Decenso por gradiente regularizado')
    plt.yscale('log')
    plt.xlabel('Iteraciones', fontsize=18)
    plt.ylabel('Norma de x', fontsize=18)
    plt.title('Evolución de la norma de x a lo largo de las iteraciones', fontsize=20)
    plt.legend(fontsize=18)
    plt.grid()
    plt.show()

def relative_error(x_iter):
    return np.linalg.norm(x_iter - x_svd) / np.linalg.norm(x_svd)

def grafico_error_relativo():
    x = x_init.copy()
    history_F = []
    history_norm_x = []
    history_residual = []

    x_reg = x_init.copy()
    history_F_reg = []
    history_norm_x_reg = []
    history_residual_reg = []

    for _ in range(iterations):
        grad = grad_F(x)
        if np.any(np.isnan(grad)) or np.any(np.isinf(grad)):
            break
        x = x - s * grad
        history_F.append(F(x))
        history_norm_x.append(np.linalg.norm(x) ** 2)
        history_residual.append(np.linalg.norm(A @ x - b) ** 2)

        grad_reg = grad_F2(x_reg, delta2)
        if np.any(np.isnan(grad)) or np.any(np.isinf(grad)):
            break
        x_reg = x_reg - s * grad_reg
        history_F_reg.append(F2(x_reg, delta2))
        history_norm_x_reg.append(np.linalg.norm(x_reg) ** 2)
        history_residual_reg.append(np.linalg.norm(A @ x_reg - b) ** 2)

    relative_errors = [relative_error(x) for x in history_F]
    relative_errors_reg = [relative_error(x) for x in history_F_reg]

    plt.figure(figsize=(10, 6))
    plt.plot(range(iterations), relative_errors, label='Error relativo (Decenso por gradiente)')
    plt.plot(range(iterations), relative_errors_reg, label='Error relativo (Decenso por gradiente regularizado)')
    plt.xlabel('Iteraciones', fontsize=18)
    plt.ylabel('Error de x relativo a x_opt', fontsize=18)
    plt.yscale('log')
    plt.title('Error relativo vs Iteraciones', fontsize=20)
    plt.legend(fontsize=18)
    plt.grid()
    plt.show()

def variacion_de_step(A, b, x0, n_iter):
    A = A / np.linalg.norm(A)
    b = b / np.linalg.norm(b)
    x0 = x0 / np.linalg.norm(x0)

    steps = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, s]
    errores = []
    for step in steps:
        x = gradient_descent_F(A, b, step, x0, n_iter)[0]
        x_reg = gradient_descent_F2(A, b, step, x0, n_iter, delta2)[0]
        errores.append([relative_error(x), relative_error(x_reg)])
    errores = np.array(errores)

    num_steps = len(steps)
    bar_width = 0.4
    index = np.arange(num_steps)
    etiquetas = ['10^-3', '10^-4', '10^-5', '10^-6', '10^-7', '10^-8', 's']
    
    fig, ax = plt.subplots(figsize=(10, 6))

    bar1 = ax.bar(index, errores[:, 0], bar_width, label='Gradiente Descendente', edgecolor='black')
    bar2 = ax.bar(index + bar_width, errores[:, 1], bar_width, label='Gradiente Descendente Regularizado', edgecolor='black')

    ax.set_xlabel('Steps', fontsize=18)
    ax.set_ylabel('Errores', fontsize=18)
    ax.set_title('Comparación de Errores para Diferentes Métodos', fontsize=20)
    ax.set_xticks(index + bar_width)
    ax.set_xticklabels(etiquetas)
    ax.legend(fontsize=18)
    
    plt.show()

def variacion_de_delta(A, b, x0, n_iter):

    S = np.linalg.svd(A)[1] # Ya vienen ordenados de mayor a menor
    deltas = []
    for i in range(len(S)):
        delta = 1e-2 * S[i]
        deltas.append(delta)
    errores = []
    for delta in deltas:
        x_reg = gradient_descent_F2(A, b, s, x0, n_iter, delta)[0]
        errores.append(relative_error(x_reg))

    x = np.arange(len(deltas))
    plt.figure()
    plt.title('Error en función de delta', fontsize=20)
    plt.bar(x, errores, tick_label=[f'$\sigma_{i}/100$' for i in range(len(deltas))])
    
    plt.ylabel('Error', fontsize=18)
    plt.xlabel('Delta', fontsize=18)
    plt.yscale('log')
    plt.xticks(rotation=10)
    
    plt.show()


grafico_costo_convergencia()
grafico_norma_convergencia()
grafico_norma_x()
grafico_error_relativo()
variacion_de_delta(A, b, x_init, iterations)
variacion_de_step(A, b, x_init, iterations)