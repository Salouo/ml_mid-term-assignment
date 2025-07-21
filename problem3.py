import numpy as np
import matplotlib.pyplot as plt
from numpy.random import default_rng

rng = default_rng(42)

# ---------------------------- 1. synthetic data ---------------------------- #
n, d = 200, 4
X = 3 * (rng.random((n, d)) - 0.5)
y = rng.choice([-1, 1], size=n)
lam = 0.5        # regularisation parameter
T   = 1000       # projected‑gradient iterations

# ---------------------------- 2. helper functions -------------------------- #
def hinge(u):
    """Standard hinge loss max(0, 1 - u)."""
    return np.maximum(0.0, 1.0 - u)

def primal_obj(w):
    """Primal objective: hinge loss + (lam/2)*||w||²."""
    return hinge(y * (X @ w)).sum() + 0.5 * lam * np.dot(w, w)

def dual_obj(alpha, K):
    """Dual objective (corrected coefficients)."""
    return -0.5 / lam * alpha @ K @ alpha + alpha.sum()

# Kernel matrix K_ij = y_i y_j x_i^T x_j
K = (y[:, None] * X) @ (y[:, None] * X).T

# ---------------------------- 3. projected gradient ------------------------ #
L   = np.linalg.eigvalsh(K).max() / lam   # Lipschitz constant of gradient
eta = 1.0 / L                             # fixed step size
alpha = np.zeros(n)

hist_P, hist_D, hist_gap = [], [], []

for _ in range(T):
    w = (alpha * y) @ X / lam            # w = (1/lam) * Σ alpha_i y_i x_i
    P = primal_obj(w)
    D = dual_obj(alpha, K)
    hist_P.append(P)
    hist_D.append(D)
    hist_gap.append(P - D)

    grad = (2.0 / lam) * (K @ alpha) - 1.0   # gradient of negative dual
    alpha = np.clip(alpha - eta * grad, 0.0, 1.0)

print(f"Final duality gap: {hist_gap[-1]:.4e}")

# ---------------------------- 4. sub‑gradient baseline --------------------- #
w_sg = np.zeros(d)
c = 1.0
hist_P_sg = []
for t in range(1, T + 1):
    idx = y * (X @ w_sg) < 1
    g   = -(y[idx, None] * X[idx]).sum(axis=0) + lam * w_sg
    lr  = c / np.sqrt(t)
    w_sg -= lr * g
    hist_P_sg.append(primal_obj(w_sg))

# ---------------------------- 5. plot results ------------------------------ #
t = np.arange(T)
plt.figure(figsize=(7, 4))
plt.semilogy(t, hist_P,          label='Primal objective')
plt.semilogy(t, hist_D,          label='Dual objective')
plt.semilogy(t, hist_gap,        label='Duality gap')
plt.semilogy(t, hist_P_sg, '--', label='Primal (sub‑gradient)')
plt.xlabel('Iteration')
plt.ylabel('Objective (log scale)')
plt.title('Primal / Dual Convergence')
plt.legend()
plt.grid(True, which='both', linestyle=':')
plt.tight_layout()
plt.show()
