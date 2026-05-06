"""
Generate dataset for steady-state Fick diffusion with smooth exponential diffusivity:

    div( D(x,y) * grad(C) ) = 0   in [0,1]^2

Boundary conditions:
    C = 1   at x=0  (left,  Dirichlet)
    C = 0   at x=1  (right, Dirichlet)
    dC/dn=0 at y=0 and y=1  (Neumann, zero normal flux)

Diffusivity field:
    D(x,y) = exp( alpha*x + beta*y )
    alpha, beta ~ Uniform(-1, 1)  drawn independently per sample

This gives D in the range [exp(-2), exp(2)] ≈ [0.14, 7.4].
Each sample has a different smooth D field, producing a different concentration
field C — making the operator learning problem meaningful:

    G : D(x,y)  -->  C(x,y)

Physical interpretation: an exponentially graded material (common in composites,
biological tissue, porous media) where diffusivity varies smoothly with position.

Coordinate convention (matches DarcyFlow PWC dataset):
    coeff[i, j, s] = D value at  (x=x_lin[i], y=y_lin[j])
    sol_fdm[i, j, s] = C value at (x=x_lin[i], y=y_lin[j])
    X[i, j] = x_lin[i],   Y[i, j] = y_lin[j]   (meshgrid with indexing='ij')
"""

import numpy as np
import h5py
import os
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve


# ── FDM solver ────────────────────────────────────────────────────────────────

def solve_fick_fdm(D, nx=29, ny=29):
    """
    Solve div(D grad C) = 0 by finite differences.

    D       : (nx, ny) array of diffusivity values
    Returns C : (nx, ny) concentration field
    """
    N = nx * ny

    def row(i, j):      # linear index: i=x-idx, j=y-idx
        return i * ny + j

    A = lil_matrix((N, N), dtype=np.float64)
    b = np.zeros(N)

    for i in range(nx):
        for j in range(ny):
            r = row(i, j)

            # ── Dirichlet BCs at x=0 and x=1 ─────────────────────────────
            if i == 0:
                A[r, r] = 1.0
                b[r]    = 1.0   # C = 1 at x=0
                continue
            if i == nx - 1:
                A[r, r] = 1.0
                b[r]    = 0.0   # C = 0 at x=1
                continue

            # ── Interface diffusivities (arithmetic mean) ─────────────────
            D_xp = 0.5 * (D[i, j] + D[i+1, j])
            D_xm = 0.5 * (D[i, j] + D[i-1, j])

            # ── Neumann at y=0 and y=1 via ghost cells ────────────────────
            if j == 0:
                D_yp = 0.5 * (D[i, j] + D[i, j+1])
                D_ym = D_yp
            elif j == ny - 1:
                D_ym = 0.5 * (D[i, j] + D[i, j-1])
                D_yp = D_ym
            else:
                D_yp = 0.5 * (D[i, j] + D[i, j+1])
                D_ym = 0.5 * (D[i, j] + D[i, j-1])

            # ── Assemble row ──────────────────────────────────────────────
            A[r, r]           = -(D_xp + D_xm + D_yp + D_ym)
            A[r, row(i+1, j)] =  D_xp
            A[r, row(i-1, j)] =  D_xm

            if j == 0:
                A[r, row(i, 1)]    = D_yp + D_ym
            elif j == ny - 1:
                A[r, row(i, ny-2)] = D_yp + D_ym
            else:
                A[r, row(i, j+1)]  = D_yp
                A[r, row(i, j-1)]  = D_ym

    C = spsolve(A.tocsr(), b)
    return C.reshape(nx, ny)


# ── Dataset generator ─────────────────────────────────────────────────────────

def generate_dataset(n_samples, nx=29, ny=29, seed=42):
    rng = np.random.default_rng(seed)

    x_lin = np.linspace(0.0, 1.0, nx)
    y_lin = np.linspace(0.0, 1.0, ny)
    X, Y  = np.meshgrid(x_lin, y_lin, indexing='ij')   # X[i,j]=x_i, Y[i,j]=y_j

    # Random parameters: alpha, beta ~ Uniform(-1, 1)
    alphas = rng.uniform(-1.0, 1.0, size=n_samples)
    betas  = rng.uniform(-1.0, 1.0, size=n_samples)

    coeff   = np.zeros((nx, ny, n_samples))
    sol_fdm = np.zeros((nx, ny, n_samples))

    for s in range(n_samples):
        D = np.exp(alphas[s] * X + betas[s] * Y)   # shape (nx, ny)
        C = solve_fick_fdm(D, nx, ny)
        coeff[:, :, s]   = D
        sol_fdm[:, :, s] = C
        if (s + 1) % 200 == 0:
            print(f'  {s+1}/{n_samples} done')

    return X, Y, coeff, sol_fdm


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    out_dir = os.path.dirname(os.path.abspath(__file__))

    print('Generating training data (1000 samples)...')
    X, Y, coeff, sol = generate_dataset(1000, seed=42)
    path = os.path.join(out_dir, 'fick_train.mat')
    with h5py.File(path, 'w') as f:
        f.create_dataset('X',       data=X,     dtype='float64')
        f.create_dataset('Y',       data=Y,     dtype='float64')
        f.create_dataset('coeff',   data=coeff, dtype='float64')
        f.create_dataset('sol_fdm', data=sol,   dtype='float64')
    print(f'  Saved to {path}')

    print('Generating test data (200 samples)...')
    X, Y, coeff, sol = generate_dataset(200, seed=999)
    path = os.path.join(out_dir, 'fick_test_in.mat')
    with h5py.File(path, 'w') as f:
        f.create_dataset('X',       data=X,     dtype='float64')
        f.create_dataset('Y',       data=Y,     dtype='float64')
        f.create_dataset('coeff',   data=coeff, dtype='float64')
        f.create_dataset('sol_fdm', data=sol,   dtype='float64')
    print(f'  Saved to {path}')

    print('Done.')
