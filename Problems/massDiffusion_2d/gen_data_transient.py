"""
Generate transient mass diffusion dataset:

    dC/dt = div( D(x,y) * grad(C) )   in [0,1]^2,  t in [0, T]

IC:  C(x,y,0) = 1-x   (linear profile matching steady-state BCs)
BCs: C=1 at x=0, C=0 at x=1, dC/dn=0 at y=0 and y=1

Diffusivity: D(x,y) = exp(alpha*x + beta*y),  alpha,beta ~ Uniform(-1,1)

Solver: Crank-Nicolson (2nd-order in time, unconditionally stable)

Storage convention (matches gen_data.py):
    coeff[i, j, s]       = D at (x=x_lin[i], y=y_lin[j])  for sample s
    sol_fdm[i, j, t, s]  = C at (x=x_lin[i], y=y_lin[j], time=t_arr[t]) for sample s
"""

import numpy as np
import h5py
import os
from scipy.sparse import lil_matrix, eye as speye
from scipy.sparse.linalg import factorized


# ── Spatial operator ──────────────────────────────────────────────────────────

def build_diffusion_operator(D, nx, ny):
    """
    Build the discrete div(D grad) stencil for interior nodes only.
    Dirichlet BC nodes (i=0, i=nx-1) are excluded from the stencil.

    Returns:
        L         : (N, N) CSR matrix  (rows for BC nodes are all-zero)
        int_idx   : list of interior node flat indices
        bc_left   : list of left-wall node indices  (C=1)
        bc_right  : list of right-wall node indices (C=0)
    """
    N = nx * ny

    def row(i, j):
        return i * ny + j

    L = lil_matrix((N, N), dtype=np.float64)

    for i in range(1, nx - 1):
        for j in range(ny):
            r = row(i, j)
            D_xp = 0.5 * (D[i, j] + D[i + 1, j])
            D_xm = 0.5 * (D[i, j] + D[i - 1, j])
            if j == 0:
                D_yp = 0.5 * (D[i, j] + D[i, j + 1])
                D_ym = D_yp
            elif j == ny - 1:
                D_ym = 0.5 * (D[i, j] + D[i, j - 1])
                D_yp = D_ym
            else:
                D_yp = 0.5 * (D[i, j] + D[i, j + 1])
                D_ym = 0.5 * (D[i, j] + D[i, j - 1])

            L[r, r]           = -(D_xp + D_xm + D_yp + D_ym)
            L[r, row(i+1, j)] =  D_xp
            L[r, row(i-1, j)] =  D_xm
            if j == 0:
                L[r, row(i, 1)]      = D_yp + D_ym
            elif j == ny - 1:
                L[r, row(i, ny - 2)] = D_yp + D_ym
            else:
                L[r, row(i, j + 1)]  = D_yp
                L[r, row(i, j - 1)]  = D_ym

    bc_left  = [row(0,      j) for j in range(ny)]
    bc_right = [row(nx - 1, j) for j in range(ny)]
    int_idx  = [r for r in range(N) if r not in set(bc_left + bc_right)]

    return L.tocsr(), int_idx, bc_left, bc_right


# ── Crank-Nicolson solver ─────────────────────────────────────────────────────

def solve_transient_cn(D, nx=29, ny=29, T=2.0, n_t=11):
    """
    Solve dC/dt = div(D grad C) with CN time-stepping.
    IC: C = 1-x.  BCs: C=1 at x=0, C=0 at x=1, Neumann at y=0/1.

    Returns sol: (nx, ny, n_t) array of concentration snapshots.
    """
    N  = nx * ny
    dt = T / (n_t - 1)

    x_lin = np.linspace(0, 1, nx)
    y_lin = np.linspace(0, 1, ny)
    X, _  = np.meshgrid(x_lin, y_lin, indexing='ij')

    # ── Initial condition ─────────────────────────────────────────────────────
    C_full = (1.0 - X).ravel().copy()   # C0 = 1 - x

    # ── Spatial operator and index sets ───────────────────────────────────────
    L, int_idx, bc_left, bc_right = build_diffusion_operator(D, nx, ny)

    # BC values (constant in time)
    C_bc           = np.zeros(N)
    C_bc[bc_left]  = 1.0
    C_bc[bc_right] = 0.0

    # Contribution of fixed BCs to interior RHS (constant every step)
    bc_contrib_int = L.dot(C_bc)[int_idx]   # (n_int,)

    # Interior sub-matrices
    n_int   = len(int_idx)
    L_int   = L[int_idx, :][:, int_idx]
    I_int   = speye(n_int, format='csr')

    LHS     = (I_int - 0.5 * dt * L_int).tocsc()
    RHS_mat =  I_int + 0.5 * dt * L_int
    bc_rhs  = dt * bc_contrib_int          # same every step

    solve_step = factorized(LHS)           # LU factorisation (reused each step)

    # ── Time march ────────────────────────────────────────────────────────────
    sol    = np.zeros((nx, ny, n_t))
    sol[:, :, 0] = C_full.reshape(nx, ny)

    C_int = C_full[int_idx].copy()

    for t_idx in range(1, n_t):
        rhs   = RHS_mat.dot(C_int) + bc_rhs
        C_int = solve_step(rhs)
        C_new = C_bc.copy()
        C_new[int_idx] = C_int
        sol[:, :, t_idx] = C_new.reshape(nx, ny)

    return sol


# ── Dataset generator ─────────────────────────────────────────────────────────

def generate_dataset(n_samples, nx=29, ny=29, T=2.0, n_t=11, seed=42):
    rng = np.random.default_rng(seed)

    x_lin  = np.linspace(0, 1, nx)
    y_lin  = np.linspace(0, 1, ny)
    X, Y   = np.meshgrid(x_lin, y_lin, indexing='ij')
    t_arr  = np.linspace(0, T, n_t)

    alphas = rng.uniform(-1.0, 1.0, size=n_samples)
    betas  = rng.uniform(-1.0, 1.0, size=n_samples)

    coeff   = np.zeros((nx, ny,      n_samples))
    sol_fdm = np.zeros((nx, ny, n_t, n_samples))

    for s in range(n_samples):
        D = np.exp(alphas[s] * X + betas[s] * Y)
        sol = solve_transient_cn(D, nx, ny, T, n_t)
        coeff[:, :,    s] = D
        sol_fdm[:, :, :, s] = sol
        if (s + 1) % 100 == 0:
            print(f'  {s + 1}/{n_samples} done')

    return X, Y, t_arr, coeff, sol_fdm


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    out_dir = os.path.dirname(os.path.abspath(__file__))
    T, n_t  = 2.0, 11   # t = 0, 0.2, 0.4, ..., 2.0

    print('Generating training data (1000 samples)...')
    X, Y, t_arr, coeff, sol = generate_dataset(1000, T=T, n_t=n_t, seed=42)
    path = os.path.join(out_dir, 'diff_transient_train.mat')
    with h5py.File(path, 'w') as f:
        f.create_dataset('X',       data=X,     dtype='float64')
        f.create_dataset('Y',       data=Y,     dtype='float64')
        f.create_dataset('t_arr',   data=t_arr, dtype='float64')
        f.create_dataset('coeff',   data=coeff, dtype='float64')
        f.create_dataset('sol_fdm', data=sol,   dtype='float64')
    print(f'  Saved to {path}')

    print('Generating test data (200 samples)...')
    X, Y, t_arr, coeff, sol = generate_dataset(200, T=T, n_t=n_t, seed=999)
    path = os.path.join(out_dir, 'diff_transient_test_in.mat')
    with h5py.File(path, 'w') as f:
        f.create_dataset('X',       data=X,     dtype='float64')
        f.create_dataset('Y',       data=Y,     dtype='float64')
        f.create_dataset('t_arr',   data=t_arr, dtype='float64')
        f.create_dataset('coeff',   data=coeff, dtype='float64')
        f.create_dataset('sol_fdm', data=sol,   dtype='float64')
    print(f'  Saved to {path}')

    print('Done.')
