"""
Generate figures for the LaTeX report.
Run from Examples/06_FickDiffusion/ :
    python gen_figures.py
"""
import sys
sys.path.append("../..")

import numpy as np
import h5py
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scienceplots
from scipy.interpolate import griddata
from torch.autograd import Variable

# ── reproducibility ──────────────────────────────────────────────────────────
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(1234)

if torch.cuda.is_available():
    device = 'cuda:0'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'

dtype = torch.float32

# ── load data ─────────────────────────────────────────────────────────────────
from Utils.utils import np2tensor

data_test = h5py.File('../../Problems/massDiffusion_2d/diff_test_in.mat', 'r')
res = 29
n_test = 200

def get_data(data, ndata, dtype, n0=0):
    a = np2tensor(np.array(data["coeff"][..., n0:n0+ndata]).T, dtype)
    u = np2tensor(np.array(data["sol_fdm"][..., n0:n0+ndata]).T, dtype)
    X, Y = np.array(data['X']).T, np.array(data['Y']).T
    mesh  = np2tensor(np.vstack([X.ravel(), Y.ravel()]).T, dtype)
    gridx = mesh.reshape(-1, 2)
    x = gridx.repeat((ndata, 1, 1))
    a = a.reshape(ndata, -1, 1)
    u = u.reshape(ndata, -1, 1)
    return a, u, x, gridx

a_test, u_test, x_test, grid_test = get_data(data_test, n_test, dtype)

# ── re-define BranchNet (required before torch.load can unpickle the model) ──
import torch.nn as nn
from Networks.CNNet import CNNet2d

class BranchNet(nn.Module):
    def __init__(self, conv_arch, fc_arch, nx_size, ny_size, dtype=None):
        super().__init__()
        self.nx_size, self.ny_size = nx_size, ny_size
        self.conv = CNNet2d(conv_arch=conv_arch, fc_arch=fc_arch,
                            activation_conv='SiLU', activation_fc='SiLU',
                            kernel_size=(3, 3), stride=2, dtype=dtype)

    def forward(self, x):
        x = x.reshape(-1, self.ny_size, self.nx_size).unsqueeze(1)
        return self.conv(x)

# ── load trained model ────────────────────────────────────────────────────────
from Solvers.PIDeepONet import PIDeepONet

solver = PIDeepONet.Solver(device=device, dtype=dtype)
model_trained = solver.loadModel(
    path='saved_models/PIDeepONetBatch_diff/',
    name='model_pideeponet_besterror')


# ── mollifier ─────────────────────────────────────────────────────────────────
class mollifier:
    def __call__(self, u, x):
        xx, yy = x[..., 0:1], x[..., 1:2]
        return u * xx * (1 - xx) * torch.sin(np.pi * yy) + (1 - xx)

# ── predict ───────────────────────────────────────────────────────────────────
x_var = Variable(x_test.to(device), requires_grad=False)
a_var = a_test.to(device)
with torch.no_grad():
    u_pred = model_trained['u'](x_var, a_var)
    u_pred = mollifier()(u_pred, x_var).cpu()

err = solver.getError(u_pred, u_test)
print(f'Test L2 relative error: {err.item():.4f}')

# ── output dir ────────────────────────────────────────────────────────────────
import os
OUT = '../../doc/figures'
os.makedirs(OUT, exist_ok=True)

grid_np = grid_test.numpy()    # (841, 2)

# ── colormap helper ───────────────────────────────────────────────────────────
mesh_plot = np.meshgrid(np.linspace(0, 1, 100), np.linspace(0, 1, 100))
xp, yp = mesh_plot[0], mesh_plot[1]

def to_contour(field_tensor):
    vals = field_tensor.numpy().ravel()
    return griddata((grid_np[:, 0], grid_np[:, 1]), vals, (xp, yp), method='cubic')

# ─────────────────────────────────────────────────────────────────────────────
# Figure 1: D, C_true, C_pred, |error| for one sample
# ─────────────────────────────────────────────────────────────────────────────
SAMPLE_IDX = 3   # pick a sample with visible spatial variation

fields = [
    a_test[SAMPLE_IDX],
    u_test[SAMPLE_IDX],
    u_pred[SAMPLE_IDX],
    torch.abs(u_test[SAMPLE_IDX] - u_pred[SAMPLE_IDX]),
]
titles = [
    r'$D(x,y)$  [diffusivity]',
    r'$C_\mathrm{true}$',
    r'$C_\mathrm{pred}$',
    r'$|C_\mathrm{true} - C_\mathrm{pred}|$',
]

with plt.style.context(['science', 'no-latex']):
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    for ax, field, title in zip(axes, fields, titles):
        z = to_contour(field)
        cf = ax.contourf(xp, yp, z, levels=14, cmap='jet')
        fig.colorbar(cf, ax=ax, shrink=0.85)
        ax.set_title(title, fontsize=14)
        ax.set_xlabel('x', fontsize=12)
        ax.set_ylabel('y', fontsize=12)
    plt.tight_layout()

fig.savefig(f'{OUT}/mass_fields.pdf', bbox_inches='tight')
fig.savefig(f'{OUT}/mass_fields.png', dpi=150, bbox_inches='tight')
print('Saved mass_fields')

# ─────────────────────────────────────────────────────────────────────────────
# Figure 2: loss curves (train / test / pde)
# ─────────────────────────────────────────────────────────────────────────────
import scipy.io
loss_saved = scipy.io.loadmat('saved_models/PIDeepONetBatch_diff/loss_pideeponet.mat')
epochs = np.arange(1, len(loss_saved['loss_pde'].ravel()) + 1)

with plt.style.context(['science', 'no-latex']):
    fig2, ax2 = plt.subplots(figsize=(7, 5))
    ax2.semilogy(epochs, loss_saved['loss_pde'].ravel(),
                 linewidth=1.8, label='PDE loss')
    ax2.semilogy(epochs, loss_saved['loss_train'].ravel(),
                 linewidth=1.8, linestyle='--', label='train loss')
    ax2.set_xlabel('Epoch', fontsize=13)
    ax2.set_ylabel('Loss', fontsize=13)
    ax2.legend(fontsize=12)
    ax2.set_title('Training Loss Convergence', fontsize=14)
    plt.tight_layout()

fig2.savefig(f'{OUT}/mass_loss.pdf', bbox_inches='tight')
fig2.savefig(f'{OUT}/mass_loss.png', dpi=150, bbox_inches='tight')
print('Saved mass_loss')

# ─────────────────────────────────────────────────────────────────────────────
# Figure 3: test L2 error vs wall-clock time
# ─────────────────────────────────────────────────────────────────────────────
time_arr  = loss_saved['time'].ravel()
error_arr = np.array(loss_saved['error']).ravel()

with plt.style.context(['science', 'no-latex']):
    fig3, ax3 = plt.subplots(figsize=(7, 5))
    ax3.semilogy(time_arr, error_arr, linewidth=1.8, color='C2', label=r'$L^2$ test error')
    ax3.set_xlabel('Wall-clock time (s)', fontsize=13)
    ax3.set_ylabel(r'Relative $L^2$ error', fontsize=13)
    ax3.legend(fontsize=12)
    ax3.set_title(r'Test Error vs Training Time', fontsize=14)
    plt.tight_layout()

fig3.savefig(f'{OUT}/mass_error.pdf', bbox_inches='tight')
fig3.savefig(f'{OUT}/mass_error.png', dpi=150, bbox_inches='tight')
print('Saved mass_error')

print('All figures saved to', os.path.abspath(OUT))
