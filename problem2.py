import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import eigvalsh

# ------------------------- helper functions ------------------------- #

def objective(w, A, mu, lam):
    """J(w) = 0.5 (w-mu)^T A (w-mu) + lam * ||w||_1"""
    quad = 0.5 * (w - mu) @ (A @ (w - mu))
    return quad + lam * np.abs(w).sum()

def grad_quadratic(w, A, mu):
    """∇ of the quadratic part only"""
    return A @ (w - mu)

def soft_threshold(v, thresh):
    """prox_{lam||.||_1}(v)"""
    return np.sign(v) * np.maximum(np.abs(v) - thresh, 0.0)

def proximal_gradient(A, mu, lam, w0, max_iter=200, alpha=1.6):
    """PG with fixed step η = 1/L"""
    L = eigvalsh(A).max()
    eta = alpha / L
    w = w0.copy()
    obj_hist = []
    for _ in range(max_iter):
        obj_hist.append(objective(w, A, mu, lam))
        g = grad_quadratic(w, A, mu)
        w = soft_threshold(w - eta * g, lam * eta)
    obj_hist.append(objective(w, A, mu, lam))
    return w, np.array(obj_hist)

def adagrad_pg(A, mu, lam, w0, max_iter=200, eta0=0.4, eps=1e-8):
    """AdaGrad-style PG: coordinate-wise adaptive steps"""
    w = w0.copy()
    G = np.zeros_like(w)          # accumulate g^2
    obj_hist = []
    for _ in range(max_iter):
        obj_hist.append(objective(w, A, mu, lam))
        g = grad_quadratic(w, A, mu)
        G += g**2
        step = eta0 / (np.sqrt(G) + eps)   # vector step
        w = soft_threshold(w - step * g, lam * step)
    obj_hist.append(objective(w, A, mu, lam))
    return w, np.array(obj_hist)

# ------------------------- Part 1: baseline A ----------------------- #

A1  = np.array([[3.0, 0.5],
                [0.5, 1.0]])
mu  = np.array([1.0, 2.0])
lam_pg = 0.1
w0  = np.zeros(2)

w_pg, J_pg = proximal_gradient(A1, mu, lam_pg, w0, max_iter=100)
J_star = J_pg[-1]
err_pg = J_pg - J_star

plt.figure()
plt.semilogy(err_pg, label='PG (λ=0.1)')
plt.xlabel('Iteration $t$')
plt.ylabel(r'$|J(w_t)-J^\star|$')
plt.title('Part1 - Proximal Gradient Convergence (semi-log)')
plt.grid(True, which='both', linestyle=':')
plt.tight_layout()
plt.legend()

# ----- sparse path λ ∈ [0.01, 10] ----------------------------------- #
lam_path = np.arange(0.01, 10.01, 0.01)
w_path = np.zeros((len(lam_path), 2))
for k, lam in enumerate(lam_path):
    w_path[k], _ = proximal_gradient(A1, mu, lam, w0, max_iter=200)

plt.figure()
plt.plot(lam_path, w_path[:, 0], label='$w_1$')
plt.plot(lam_path, w_path[:, 1], label='$w_2$')
plt.gca().invert_xaxis()
plt.xlabel(r'$\lambda$')
plt.ylabel('Optimal coordinate value')
plt.title('Part1 - Lasso solution path (semi-log)')
plt.grid(True, linestyle=':')
plt.legend()
plt.tight_layout()

# ------------------------- Part 2: new A ---------------------------- #

A2 = np.array([[300.0, 0.5],
               [0.5,  10.0]])
lam2 = 0.8
w0   = np.zeros(2)

w_pg2, J_pg2 = proximal_gradient(A2, mu, lam2, w0, max_iter=400)
w_ag2, J_ag2 = adagrad_pg(A2, mu, lam2, w0, max_iter=400)

J_star2 = min(J_pg2[-1], J_ag2[-1])
err_pg2 = J_pg2 - J_star2
err_ag2 = J_ag2 - J_star2

plt.figure()
plt.semilogy(err_pg2, label='PG')
plt.semilogy(err_ag2, label='AdaGrad-PG (semi-log)')
plt.xlabel('Iteration $t$')
plt.ylabel(r'$|J(w_t)-J^\star|$')
plt.title('Part2 - PG vs AdaGrad-PG')
plt.grid(True, which='both', linestyle=':')
plt.legend()
plt.tight_layout()
plt.show()
