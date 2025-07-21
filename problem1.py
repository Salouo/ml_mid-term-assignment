import numpy as np
import matplotlib.pyplot as plt
from numpy.random import rand, randn

# -------------------------  helper functions  ------------------------- #

def sigmoid(z):
    """Numerically stable sigmoid."""
    pos = z >= 0
    neg = ~pos
    out = np.empty_like(z, dtype=float)
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[neg])
    out[neg] = ez / (1.0 + ez)
    return out

def objective(w, X, y, lam):
    """L2-regularised logistic-loss."""
    return np.logaddexp(0, -y * (X @ w)).sum() + 0.5 * lam * np.dot(w, w)

def grad(w, X, y, lam):
    s = sigmoid(-y * (X @ w))
    return -(y * s) @ X + lam * w

def hessian(w, X, y, lam):
    s = sigmoid(y * (X @ w))
    W = s * (1.0 - s)             # n-dimensional vector
    H = X.T * W @ X               # == X.T @ diag(W) @ X
    H += lam * np.eye(X.shape[1])
    return H

def gradient_descent(X, y, lam, w0, max_iter=200, alpha=1):
    # Lipschitz constant of ∇J
    L = 0.25 * np.linalg.eigvalsh(X.T @ X).max() + lam
    eta = 1.0 / L
    w = w0.copy()
    obj_hist = []
    for _ in range(max_iter):
        obj_hist.append(objective(w, X, y, lam))
        g = grad(w, X, y, lam)
        w -= eta * g * alpha

    obj_hist.append(objective(w, X, y, lam))
    return w, np.array(obj_hist)

def newton_method(X, y, lam, w0, max_iter=200, alpha=0.18):
    w = w0.copy()
    obj_hist = []
    for _ in range(max_iter):
        obj_hist.append(objective(w, X, y, lam))

        g = grad(w, X, y, lam)
        H = hessian(w, X, y, lam)
        p = np.linalg.solve(H, g)

        w -= alpha * p           

    obj_hist.append(objective(w, X, y, lam))
    return w, np.array(obj_hist)

# ------------------------------  run  -------------------------------- #
n = 200
X = 3 * (rand(n, 4) - 0.5)
y = (2 * X[:, 1] - 1 * X[:,2] + 0.5 + 0.5 * randn(n)) > 0
y = 2 * y -1


lam = 1.0
max_iter = 200

d = X.shape[1]
w0 = np.zeros(d)

w_gd, Jg = gradient_descent(X, y, lam, w0, max_iter)
print(f"[GD]  final J = {Jg[-1]:.6e}, iters = {len(Jg)-1}")

w_nt, Jn = newton_method(X, y, lam, w0, max_iter)
print(f"[NT]  final J = {Jn[-1]:.6e}, iters = {len(Jn)-1}")

J_hat = min(Jg[-1], Jn[-1])
err_gd = Jg - J_hat
err_nt = Jn - J_hat

plt.semilogy(err_gd, label='Gradient Descent')
plt.semilogy(err_nt, label='Newton')
plt.xlabel('Iteration $t$')
plt.ylabel(r'$|J(w_t) - J(\hat{w})|$')
plt.title('Convergence Comparison (semi-log)')
plt.legend()
plt.grid(True, which='both', linestyle=':')
plt.tight_layout()
plt.show()
