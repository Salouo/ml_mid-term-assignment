# ------------------ problem3_pg_correct.py ------------------
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import default_rng
rng = default_rng(42)

# 1. synthetic data -------------------------------------------------
n, d = 200, 4
X = 3 * (rng.random((n, d)) - 0.5)
y = rng.choice([-1, 1], size=n)
lam = 0.5              # regularisation λ
T   = 1000             # PG iterations

# 2. helper functions ----------------------------------------------
def hinge(u): return np.maximum(0.0, 1.0 - u)
def primal_obj(w):
    return hinge(y * (X @ w)).sum() + 0.5 * lam * np.dot(w, w)
def dual_obj(alpha, K):
    return -0.25 / lam * alpha @ K @ alpha + alpha.sum()

# kernel matrix  K_ij = y_i y_j x_i^T x_j
K = (y[:, None] * X) @ (y[:, None] * X).T

# 3. projected gradient on negative dual ---------------------------
L   = 2 * np.linalg.eigvalsh(K).max() / lam     # Lipschitz const.
eta = 1.0 / L
alpha = np.zeros(n)

hist_P, hist_D, hist_gap = [], [], []

for _ in range(T):
    w = (alpha * y) @ X / lam                   # w = 1/λ Σ α_i y_i x_i
    P = primal_obj(w)
    D = dual_obj(alpha, K)
    hist_P.append(P); hist_D.append(D); hist_gap.append(P - D)

    grad = (2 / lam) * (K @ alpha) - 1.0        # ∇f = 2/λ Kα - 1
    alpha = np.clip(alpha - eta * grad, 0.0, 1.0)

print(f"Final duality gap: {hist_gap[-1]:.4e}")

# 4. optional: primal sub-gradient baseline ------------------------
w_sg = np.zeros(d)
c = 1.0
hist_P_sg = []
for t in range(1, T + 1):
    idx = y * (X @ w_sg) < 1
    g   = -(y[idx, None] * X[idx]).sum(axis=0) + lam * w_sg
    lr  = c / np.sqrt(t)
    w_sg -= lr * g
    hist_P_sg.append(primal_obj(w_sg))

# 5. plot -----------------------------------------------------------
t = np.arange(T)
plt.figure(figsize=(7, 4))
plt.semilogy(t, hist_P,           label='Primal $P(w^{(t)})$')
plt.semilogy(t, hist_D,           label='Dual $D(\\alpha^{(t)})$')
plt.semilogy(t, hist_gap,         label='Gap $|P-D|$')
plt.semilogy(t, hist_P_sg, '--',  label='Primal (subgrad)')
plt.xlabel('Iteration $t$')
plt.ylabel('Objective (log scale)')
plt.title('Primal / dual convergence')
plt.legend()
plt.grid(True, which='both', linestyle=':')
plt.tight_layout()
plt.show()
# --------------------------------------------------------------
