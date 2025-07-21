import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
from numpy.random import default_rng

rng = default_rng(42)

# --------------------------- 1. toy data --------------------------- #
n, d = 200, 4
X = 3 * (rng.random((n, d)) - 0.5)
y = rng.choice([-1, 1], size=n)

lam = 0.5        # ℓ₁‑regularisation parameter
T   = 900        # prox‑sub‑gradient iterations

# --------------------------- 2. LP (ECOS) -------------------------- #
w_lp = cp.Variable(d)
xi    = cp.Variable(n)     # slack variables
e     = cp.Variable(d)     # absolute‑value helper |w| ≤ e

constr = [
    cp.multiply(y, X @ w_lp) + xi >= 1,   # margin constraints
    xi >= 0,
    w_lp <= e,
    -w_lp <= e,
    e >= 0
]
objective = cp.Minimize(cp.sum(xi) + lam * cp.sum(e))
prob = cp.Problem(objective, constr)
prob.solve(solver='ECOS')

J_LP = prob.value
print(f"LP optimum: {J_LP:.6f}")

# --------------------------- 3. PSG solver ------------------------- #
def hinge(u):
    """Vectorised hinge loss max(0, 1 ‑ u)."""
    return np.maximum(0.0, 1.0 - u)

def soft_thresh(v, tau):
    """Soft‑thresholding operator (prox of τ‖·‖₁)."""
    return np.sign(v) * np.maximum(np.abs(v) - tau, 0.0)

def primal_L1(w):
    """Primal ℓ₁‑SVM objective at weight vector w."""
    return hinge(y * (X @ w)).sum() + lam * np.abs(w).sum()

w = np.zeros(d)
hist_PSG = []
c = 1.0                 # step‑size constant (1/√t schedule)

for t in range(1, T + 1):
    u = y * (X @ w)
    g = -(y[u < 1][:, None] * X[u < 1]).sum(axis=0)   # hinge sub‑gradient
    lr = c / np.sqrt(t)                               # learning rate
    w  = soft_thresh(w - lr * g, lr * lam)            # prox update
    hist_PSG.append(primal_L1(w))

# --------------------------- 4. plotting --------------------------- #
plt.figure(figsize=(6, 4))
plt.semilogy(range(T), hist_PSG, label='Primal (PSG)')
plt.axhline(J_LP, color='k', ls='--', label='LP optimum')
plt.xlabel('Iteration')
plt.ylabel('Objective (semi‑log scale)')
plt.title('PSG convergence vs. LP optimum')
plt.legend()
plt.grid(True, which='both', linestyle=':')
plt.tight_layout()
plt.show()
