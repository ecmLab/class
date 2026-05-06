"""
Generate transient mass diffusion dataset with variable boundary conditions:

    dC/dt = lap(C)   in [0,1]^2,  t in [0, T]    (D=1, heat equation)

IC:  C(x,y,0) = 0
BCs (ramp-on to avoid corner singularity):
    C(0,y,t) = C_L * (1 - exp(-t/tau))   left wall  (Dirichlet, ramps from 0)
    C(1,y,t) = C_R * (1 - exp(-t/tau))   right wall (Dirichlet, ramps from 0)
    dC/dn = 0   at y=0 and y=1           (Neumann, fixed)

Parameters:  C_L, C_R ~ Uniform(0, 1)  drawn independently per sample.

Operator:  G : (C_L, C_R) --> C(x, y, t)

Solver: Crank-Nicolson.  Since D=1 for all samples, L is built once and
        factorised once per sample (only the RHS BC vector changes in time).
"""

import numpy as np
import h5py
import os
from scipy.sparse import lil_matrix, eye as speye
from scipy.sparse.linalg import factorized


TAU = 0.2   # ramp timescale (same constant used in the mollifier)


# ── Build D=1 diffusion operator ─────────────────────────────────────────────

def build_laplacian(nx, ny):
    """
    Discrete Laplacian for the interior nodes (D=1, uniform grid).
    Returns L (CSR), int_idx, bc_left, bc_right.
    """
    N = nx * ny

    def row(i, j):
        return i * ny + j

    L = lil_matrix((N, N), dtype=np.float64)

    for i in range(1, nx - 1):
        for j in range(ny):
            r = row(i, j)
            # uniform D=1 → all interface diffusivities = 1
            if j == 0:
                L[r, r]           = -4.0
                L[r, row(i+1, j)] =  1.0
                L[r, row(i-1, j)] =  1.0
                L[r, row(i, 1)]   =  2.0    # ghost-cell Neumann
            elif j == ny - 1:
                L[r, r]              = -4.0
                L[r, row(i+1, j)]   =  1.0
                L[r, row(i-1, j)]   =  1.0
                L[r, row(i, ny-2)]  =  2.0  # ghost-cell Neumann
            else:
                L[r, r]           = -4.0
                L[r, row(i+1, j)] =  1.0
                L[r, row(i-1, j)] =  1.0
                L[r, row(i, j+1)] =  1.0
                L[r, row(i, j-1)] =  1.0

    bc_left  = [row(0,      j) for j in range(ny)]
    bc_right = [row(nx - 1, j) for j in range(ny)]
    int_idx  = [r for r in range(N) if r not in set(bc_left + bc_right)]

    return L.tocsr(), int_idx, bc_left, bc_right


# ── Crank-Nicolson solver with time-varying BCs ───────────────────────────────

def solve_bc_transient_cn(C_L_val, C_R_val,
                           L, int_idx, bc_left, bc_right,
                           nx=29, ny=29, T=2.0, n_t=11, tau=TAU):
    """
    Solve dC/dt = lap(C) from IC=0 with ramp-on Dirichlet BCs.
    L, int_idx, bc_left, bc_right are pre-built (D=1, reused across samples).
    Returns sol: (nx, ny, n_t).
    """
    N  = nx * ny
    dt = T / (n_t - 1)
    t_arr = np.linspace(0, T, n_t)

    n_int  = len(int_idx)
    L_int  = L[int_idx, :][:, int_idx]
    I_int  = speye(n_int, format='csr')
    LHS    = (I_int - 0.5 * dt * L_int).tocsc()
    RHS_mat = I_int + 0.5 * dt * L_int
    solve_step = factorized(LHS)

    def bc_vec(t_val):
        bc = np.zeros(N)
        h  = 0.0 if t_val == 0.0 else (1.0 - np.exp(-t_val / tau))
        for idx in bc_left:
            bc[idx] = C_L_val * h
        for idx in bc_right:
            bc[idx] = C_R_val * h
        return bc

    sol   = np.zeros((nx, ny, n_t))   # t=0: all zeros (IC)
    C_int = np.zeros(n_int)           # IC at interior nodes

    for t_idx in range(1, n_t):
        t_n   = t_arr[t_idx - 1]
        t_np1 = t_arr[t_idx]

        bc_n   = bc_vec(t_n)
        bc_np1 = bc_vec(t_np1)
        bc_rhs = 0.5 * dt * (L.dot(bc_n)[int_idx] + L.dot(bc_np1)[int_idx])

        C_int = solve_step(RHS_mat.dot(C_int) + bc_rhs)

        C_new = bc_np1.copy()
        C_new[int_idx] = C_int
        sol[:, :, t_idx] = C_new.reshape(nx, ny)

    return sol


# ── Dataset generator ─────────────────────────────────────────────────────────

def generate_dataset(n_samples, nx=29, ny=29, T=2.0, n_t=11, tau=TAU, seed=42):
    rng = np.random.default_rng(seed)

    x_lin = np.linspace(0, 1, nx)
    y_lin = np.linspace(0, 1, ny)
    X, Y  = np.meshgrid(x_lin, y_lin, indexing='ij')
    t_arr = np.linspace(0, T, n_t)

    # Build D=1 operator once for all samples
    L, int_idx, bc_left, bc_right = build_laplacian(nx, ny)

    C_Ls = rng.uniform(0.0, 1.0, size=n_samples)
    C_Rs = rng.uniform(0.0, 1.0, size=n_samples)

    sol_fdm = np.zeros((nx, ny, n_t, n_samples))

    for s in range(n_samples):
        sol = solve_bc_transient_cn(
            C_Ls[s], C_Rs[s], L, int_idx, bc_left, bc_right,
            nx, ny, T, n_t, tau)
        sol_fdm[:, :, :, s] = sol
        if (s + 1) % 200 == 0:
            print(f'  {s + 1}/{n_samples} done')

    return X, Y, t_arr, C_Ls, C_Rs, sol_fdm


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    out_dir = os.path.dirname(os.path.abspath(__file__))
    T, n_t  = 2.0, 11

    print('Generating training data (1000 samples)...')
    X, Y, t_arr, C_Ls, C_Rs, sol = generate_dataset(1000, T=T, n_t=n_t, seed=42)
    path = os.path.join(out_dir, 'diff_bc_train.mat')
    with h5py.File(path, 'w') as f:
        f.create_dataset('X',       data=X,     dtype='float64')
        f.create_dataset('Y',       data=Y,     dtype='float64')
        f.create_dataset('t_arr',   data=t_arr, dtype='float64')
        f.create_dataset('C_L',     data=C_Ls,  dtype='float64')
        f.create_dataset('C_R',     data=C_Rs,  dtype='float64')
        f.create_dataset('sol_fdm', data=sol,   dtype='float64')
    print(f'  Saved to {path}')

    print('Generating test data (200 samples)...')
    X, Y, t_arr, C_Ls, C_Rs, sol = generate_dataset(200, T=T, n_t=n_t, seed=999)
    path = os.path.join(out_dir, 'diff_bc_test_in.mat')
    with h5py.File(path, 'w') as f:
        f.create_dataset('X',       data=X,     dtype='float64')
        f.create_dataset('Y',       data=Y,     dtype='float64')
        f.create_dataset('t_arr',   data=t_arr, dtype='float64')
        f.create_dataset('C_L',     data=C_Ls,  dtype='float64')
        f.create_dataset('C_R',     data=C_Rs,  dtype='float64')
        f.create_dataset('sol_fdm', data=sol,   dtype='float64')
    print(f'  Saved to {path}')

    print('Done.')
