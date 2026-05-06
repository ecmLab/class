%% generate_heat_training_data.m
% Generates training and test data for the PIDeepONet steady-state heat equation
% notebook (PIDeepONet_heatEquation_steadyState.ipynb).
%
% PDE:  -Lap(T) = q(x,y),   (x,y) in [0,1]^2
%       T = 0 on all four walls  (homogeneous Dirichlet)
%
% What is generated
% -----------------
%   heat_train.mat   : q-field samples only.
%                      sol_fdm is zeros (unused -- model is purely PI, w_data=0).
%   heat_test.mat    : q-field samples + FDM solutions T(x,y).
%                      Used to evaluate prediction accuracy after training.
%
% q-field model: zero-mean Gaussian random field (GRF)
%   k_SE(r) = exp( -r^2 / (2*l^2) )
%   q may be positive (heat source) or negative (heat sink).
%   Unlike diffusivity, q does not need to be strictly positive.
%
% FDM solver: 5-point Laplacian stencil, sparse direct solve (MATLAB backslash).
%   Each sample requires one sparse factorisation of an (n x n)^2 system
%   where n = res-2 = 27 (interior nodes per direction), system size 729x729.
%
% Array layout (h5py-compatible transposition convention)
% -------------------------------------------------------
%   The Python notebook loads data via h5py and applies .T (numpy transpose).
%   To ensure correct axis ordering after that transposition, each 2-D field
%   F(iy,ix) is stored as F' (MATLAB transpose) in the .mat file:
%
%       coeff(:,:,k)   = q_field(:,:,k)'
%       sol_fdm(:,:,k) = T_field(:,:,k)'
%       X_store        = X_mesh'
%       Y_store        = Y_mesh'
%
%   where q_field(iy,ix,k) follows the meshgrid convention X_mesh(iy,ix)=x_vals(ix).
%   After the Python .T call, fun_q can look up q with flat index iy*res + ix. 

clear; clc; rng(1234);

%% ── Configuration ──────────────────────────────────────────────────────────
res     = 29;        % grid resolution (matches notebook default)
n_train = 1000;      % number of training samples
n_test  = 200;       % number of test samples (FDM solve required)
GRF_l   = 0.20;      % GRF length scale [0,1] -- smaller = more spatial variation
GRF_sig = 1.0;       % GRF standard deviation before applying to q
out_dir = '../../Problems/heatEquation_2d';

if ~exist(out_dir, 'dir'), mkdir(out_dir); end

%% ── Grid ───────────────────────────────────────────────────────────────────
x_vals  = linspace(0, 1, res);   % x-coordinates, size (1, res)
y_vals  = linspace(0, 1, res);   % y-coordinates, size (1, res)
[X_mesh, Y_mesh] = meshgrid(x_vals, y_vals);
%   X_mesh(iy, ix) = x_vals(ix)
%   Y_mesh(iy, ix) = y_vals(iy)

h  = x_vals(2) - x_vals(1);     % uniform grid spacing (dx = dy = h)
n  = res - 2;                    % number of interior nodes per direction
N  = n^2;                        % total interior DOFs

fprintf('Grid: %d x %d   h = %.4f   Interior DOFs: %d\n', res, res, h, N);

%% ── GRF covariance -- precompute Cholesky once ─────────────────────────────
% Points: (x,y) at all res^2 nodes, in column-major order (MATLAB default).
% pts(k,:) = [x_vals(ix), y_vals(iy)] where k = (ix-1)*res + iy.
pts  = [X_mesh(:), Y_mesh(:)];                   % (res^2, 2)
n_pts = size(pts, 1);

fprintf('Building GRF covariance (%d x %d) -- one-time cost ...\n', n_pts, n_pts);

% Squared Euclidean distances (no toolbox required)
%   ||a - b||^2 = ||a||^2 + ||b||^2 - 2 a.b
sq_norms = sum(pts.^2, 2);                        % (n_pts, 1)
D2       = sq_norms + sq_norms' - 2*(pts * pts'); % (n_pts, n_pts)
D2       = max(D2, 0);                            % clamp numerical negatives

K        = exp(-D2 / (2 * GRF_l^2));             % squared-exponential kernel
K        = K + 1e-8 * eye(n_pts);                % jitter for numerical stability
L_chol   = chol(K, 'lower');                     % lower Cholesky factor

fprintf('  Done.  Cholesky factor: %d x %d\n', size(L_chol));

%% ── Sample q-fields ─────────────────────────────────────────────────────────
n_total  = n_train + n_test;
fprintf('\nSampling %d GRF realisations ...\n', n_total);

Z        = L_chol * randn(n_pts, n_total);        % (n_pts, n_total)
Z        = GRF_sig * Z;

% Reshape: q_all(iy, ix, k) -- column-major reshape in MATLAB fills iy first
%   because X_mesh(:) lists iy=1..res for ix=1, then iy=1..res for ix=2, etc.
%   So Z(iy + (ix-1)*res, k) is the value at (x_vals(ix), y_vals(iy)).
%   We want q_all(iy, ix, k), so reshape(Z, res, res, n_total) does NOT
%   give q_all(iy,ix,k) directly -- we need to permute.
Z_3d     = reshape(Z, [res, res, n_total]);       % Z_3d(iy_ord, ix_ord, k)
% In MATLAB column-major reshape: first index (iy_ord) matches the ordering
% of pts(:,1), which goes iy=1..res for each ix block.  So:
%   Z_3d(j, i, k) = Z((i-1)*res + j, k) = value at (x_vals(i), y_vals(j))
% i.e., Z_3d(iy, ix, k) = q at (x_vals(ix), y_vals(iy))  -- correct layout.
q_all    = Z_3d;                                  % (res, res, n_total): q_all(iy,ix,k)

q_train  = q_all(:, :, 1:n_train);
q_test   = q_all(:, :, n_train+1:end);

fprintf('  q range: [%.3f, %.3f]\n', min(q_all(:)), max(q_all(:)));

%% ── FDM solver: -Lap(T) = q with T=0 on all walls ──────────────────────────
% Interior nodes ordered as: flat index k = (ix_int-1)*n + iy_int  (1-indexed)
% using the Kronecker product structure of the 5-point Laplacian.
%
% A = (1/h^2) * kron(I_n, Tx) + (1/h^2) * kron(Tx, I_n)
% where Tx = tridiag(-1, 2, -1) is the 1-D negative-Laplacian on n interior nodes.
%
% System: A * T_vec = q_vec

e   = ones(n, 1);
Tx  = (1/h^2) * spdiags([-e, 2*e, -e], [-1, 0, 1], n, n);  % 1-D neg. Laplacian
In  = speye(n);
A   = kron(In, Tx) + kron(Tx, In);                           % 2-D neg. Laplacian

% Factorise once -- A is the same for all samples (uniform Dirichlet BCs, const k)
fprintf('\nFactorising FDM system matrix (%d x %d, %d nnz) ...\n', N, N, nnz(A));
[A_L, A_U, A_P, A_Q] = lu(A);   % PAQ = LU  (permuted LU for sparse A)
fprintf('  Done.\n');

%% Solve for each test sample
fprintf('\nSolving %d FDM problems for test set ...\n', n_test);
T_test = zeros(res, res, n_test);

for k = 1:n_test
    if mod(k, 20) == 0 || k == 1
        fprintf('  Sample %3d / %d\n', k, n_test);
    end

    % RHS: q at interior nodes.
    % kron(I_n, Tx) uses ordering: outer index = ix_int, inner index = iy_int.
    % q_interior(iy_int, ix_int) = q_test(iy_int+1, ix_int+1, k)
    % Column-major reshape: q_vec((ix_int-1)*n + iy_int) = q_interior(iy_int, ix_int)
    q_interior = q_test(2:res-1, 2:res-1, k);          % (n, n)
    q_vec      = q_interior(:);                         % (N, 1), column-major

    % Solve A * T_vec = q_vec  using pre-factorised LU:  Q * (U \ (L \ (P * q_vec)))
    T_vec = A_Q * (A_U \ (A_L \ (A_P * q_vec)));

    % Reconstruct full res x res field (boundary stays at zero)
    T_interior     = reshape(T_vec, [n, n]);            % T_interior(iy_int, ix_int)
    T_full         = zeros(res, res);
    T_full(2:res-1, 2:res-1) = T_interior;
    T_test(:, :, k) = T_full;
end

fprintf('  T range (test): [%.4f, %.4f]\n', min(T_test(:)), max(T_test(:)));

% Quick sanity check: all-wall BCs
assert(all(T_test(:,   1,   :) == 0, 'all'), 'Left wall BC violated');
assert(all(T_test(:,   res, :) == 0, 'all'), 'Right wall BC violated');
assert(all(T_test(1,   :,   :) == 0, 'all'), 'Bottom wall BC violated');
assert(all(T_test(res, :,   :) == 0, 'all'), 'Top wall BC violated');
fprintf('  Dirichlet BCs: OK\n');

%% ── Save -- apply transposition convention for h5py compatibility ───────────
% Python loads: data['coeff'][..., k].T  -->  (res, res) array in C-order
% To get q_all(iy, ix, k) after that .T, we must store coeff(:,:,k) = q_all(:,:,k)'
%
% Same rule for sol_fdm, X, Y.

X_store = X_mesh';          % X_store(ix, iy) = x_vals(ix)  [stores as (res,res)]
Y_store = Y_mesh';

% ── Training file (sol_fdm = zeros, never used when w_data=0) ──────────────
coeff_train  = zeros(res, res, n_train);
sol_train    = zeros(res, res, n_train);
for k = 1:n_train
    coeff_train(:, :, k) = q_train(:, :, k)';   % transpose per convention
end
train_path = fullfile(out_dir, 'heat_train.mat');
save(train_path, 'coeff_train', 'sol_train', 'X_store', 'Y_store', '-v7.3');
% Rename variables to match expected keys
m = matfile(train_path, 'Writable', true);
m.coeff   = coeff_train;
m.sol_fdm = sol_train;
m.X       = X_store;
m.Y       = Y_store;
fprintf('\nSaved: %s\n', train_path);

% ── Test file ───────────────────────────────────────────────────────────────
coeff_test_store = zeros(res, res, n_test);
sol_test_store   = zeros(res, res, n_test);
for k = 1:n_test
    coeff_test_store(:, :, k) = q_test(:, :, k)';
    sol_test_store(:, :, k)   = T_test(:, :, k)';
end
test_path = fullfile(out_dir, 'heat_test.mat');
save(test_path, '-v7.3');   % placeholder save
m2 = matfile(test_path, 'Writable', true);
m2.coeff   = coeff_test_store;
m2.sol_fdm = sol_test_store;
m2.X       = X_store;
m2.Y       = Y_store;
fprintf('Saved: %s\n', test_path);

%% ── Optional: plot a sample to verify ──────────────────────────────────────
k_show = 1;
figure('Name', 'Data verification', 'NumberTitle', 'off');

subplot(1, 2, 1);
contourf(X_mesh, Y_mesh, q_test(:,:,k_show), 40, 'LineStyle', 'none');
colorbar; axis equal tight;
title(sprintf('q(x,y) -- sample %d', k_show));
xlabel('x'); ylabel('y');

subplot(1, 2, 2);
contourf(X_mesh, Y_mesh, T_test(:,:,k_show), 40, 'LineStyle', 'none');
colorbar; axis equal tight;
title(sprintf('T(x,y) -- FDM solution at sample %d', k_show));
xlabel('x'); ylabel('y');

sgtitle('-Lap(T) = q,  T=0 on all walls');

fprintf('\nDone. Files saved to: %s\n', out_dir);
fprintf('  heat_train.mat  -- %d q-field samples (labels = zeros)\n', n_train);
fprintf('  heat_test.mat   -- %d q-field samples + FDM T solutions\n', n_test);
fprintf('\nUpdate the notebook data path:\n');
fprintf('  heat_train.mat -> ''.../heatEquation_2d/heat_train.mat''\n');
fprintf('  heat_test.mat  -> ''.../heatEquation_2d/heat_test.mat''\n');
