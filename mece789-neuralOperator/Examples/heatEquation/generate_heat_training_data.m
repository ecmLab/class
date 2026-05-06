%% generate_convdiff_data.m
% Generates training and test data for PIDeepONet_convDiff_2d.ipynb.
%
% PDE:  -k*Lap(T) + h(x,y)*T = 0,   (x,y) in [0,1]^2,   k = 1
%
% Physical interpretation:
%   Steady-state heat conduction with spatially varying convective cooling.
%   Models a hot-gas-side wall with non-uniform coolant heat transfer
%   coefficient -- directly relevant to regenerative cooling in rocket engines.
%
% Boundary conditions:
%   Left   x=0 : T = 1   (hot gas wall, Dirichlet)
%   Right  x=1 : T = 0   (coolant side, Dirichlet)
%   Top/bottom : dT/dn=0  (symmetry, Neumann)
%
% Mollifier (notebook): T = u_NN * x*(1-x)*sin(pi*y) + (1-x)
% BCs satisfied: T(0,y)=1, T(1,y)=0, dT/dy=0 at y=0,1
%
% h-field parameterisation:
%   h(x,y) = exp(alpha*sin(pi*x) + beta*cos(pi*y)),  alpha,beta~Uniform(-1,1)
%   Always positive, spatially smooth, nonlinear structure.
%
% FDM solver:
%   PDE: -Lap(T) + h(x,y)*T = 0
%   Rearranged: Lap(T) = h(x,y)*T
%   5-point Laplacian on interior nodes + diagonal reaction term.
%   System: (A_lap + H_diag) * T_vec = rhs_bc
%   where A_lap = negative Laplacian (constant), H_diag = diag(h) (per-sample).
%   Since H_diag changes per sample, we factorize per sample.
%
% Array layout: saved as (n, res, res) -- no transposition needed.
%   Python loads: np.array(data['coeff'][k]) --> (res, res) directly.

clear; clc; rng(1234);

%% ── Configuration ────────────────────────────────────────────────────────────
res     = 29;
n_train = 1000;
n_test  = 200;
out_dir = '../../Problems/convDiff_2d';

% Delete any existing files in the output directory to avoid corruption
if exist(out_dir, 'dir')
    delete(fullfile(out_dir, '*.mat'));
    fprintf('Cleared existing .mat files from %s\n', out_dir);
end
if ~exist(out_dir, 'dir'), mkdir(out_dir); end

%% ── Grid ─────────────────────────────────────────────────────────────────────
x_vals = linspace(0, 1, res);
y_vals = linspace(0, 1, res);
[X_mesh, Y_mesh] = meshgrid(x_vals, y_vals);   % X_mesh(iy, ix), Y_mesh(iy, ix)
dh = x_vals(2) - x_vals(1);                    % grid spacing

fprintf('Grid: %d x %d   h = %.4f\n', res, res, dh);

%% ── Sample h fields ──────────────────────────────────────────────────────────
n_total   = n_train + n_test;
alpha_all = 2*rand(1, n_total) - 1;   % Uniform(-1, 1)
beta_all  = 2*rand(1, n_total) - 1;

fprintf('Assembling %d h fields: h(x,y) = exp(alpha*sin(pi*x) + beta*cos(pi*y)) ...\n', n_total);
h_all = zeros(res, res, n_total);   % h_all(iy, ix, k)
for k = 1:n_total
    h_all(:,:,k) = exp(alpha_all(k)*sin(pi*X_mesh) + beta_all(k)*cos(pi*Y_mesh));
end
fprintf('  h range: [%.4f, %.4f]\n', min(h_all(:)), max(h_all(:)));

%% ── FDM solver ────────────────────────────────────────────────────────────────
% DOF ordering:
%   n_x = res-2  (interior x-nodes, excluding Dirichlet walls)
%   n_y = res    (all y-nodes, Neumann top/bottom via ghost cells)
%   N   = n_x * n_y
%   Flat index (1-based): k = (ix_int-1)*n_y + iy
%   where ix_int = ix - 1  (ix=2..res-1 -> ix_int=1..n_x)
%
% FDM system: (-Lap + H_diag) * T = rhs
%   -Lap: 5-point stencil with Neumann ghost cells at iy=1 and iy=n_y
%   H_diag: diagonal matrix with h(iy, ix) per node
%   rhs: Dirichlet BC contribution from left wall (T=1)

n_x = res - 2;
n_y = res;
N   = n_x * n_y;

% Build the constant negative Laplacian part (off-diagonal and center from
% the stencil, WITHOUT the h diagonal -- that's added per sample)
fprintf('\nBuilding constant Laplacian structure ...\n');

rows_lap = zeros(5*N, 1);
cols_lap = zeros(5*N, 1);
vals_lap = zeros(5*N, 1);
nnz_c = 0;

for ix_int = 1:n_x
    for iy = 1:n_y
        k = (ix_int-1)*n_y + iy;

        % Center: 4/h^2 (h-diagonal added per-sample separately)
        nnz_c = nnz_c+1;
        rows_lap(nnz_c)=k; cols_lap(nnz_c)=k; vals_lap(nnz_c)=4/dh^2;

        % x-left
        if ix_int > 1
            nnz_c=nnz_c+1;
            rows_lap(nnz_c)=k; cols_lap(nnz_c)=(ix_int-2)*n_y+iy;
            vals_lap(nnz_c)=-1/dh^2;
        end
        % x-right
        if ix_int < n_x
            nnz_c=nnz_c+1;
            rows_lap(nnz_c)=k; cols_lap(nnz_c)=ix_int*n_y+iy;
            vals_lap(nnz_c)=-1/dh^2;
        end
        % y-down (Neumann ghost at iy=1: T(0)=T(2), double iy+1)
        if iy > 1
            nnz_c=nnz_c+1;
            rows_lap(nnz_c)=k; cols_lap(nnz_c)=(ix_int-1)*n_y+(iy-1);
            vals_lap(nnz_c)=-1/dh^2;
        else
            nnz_c=nnz_c+1;
            rows_lap(nnz_c)=k; cols_lap(nnz_c)=(ix_int-1)*n_y+(iy+1);
            vals_lap(nnz_c)=-1/dh^2;
        end
        % y-up (Neumann ghost at iy=n_y: T(n_y+1)=T(n_y-1), double iy-1)
        if iy < n_y
            nnz_c=nnz_c+1;
            rows_lap(nnz_c)=k; cols_lap(nnz_c)=(ix_int-1)*n_y+(iy+1);
            vals_lap(nnz_c)=-1/dh^2;
        else
            nnz_c=nnz_c+1;
            rows_lap(nnz_c)=k; cols_lap(nnz_c)=(ix_int-1)*n_y+(iy-1);
            vals_lap(nnz_c)=-1/dh^2;
        end
    end
end
A_lap = sparse(rows_lap(1:nnz_c), cols_lap(1:nnz_c), vals_lap(1:nnz_c), N, N);
fprintf('  A_lap: %d x %d, %d nnz\n', N, N, nnz(A_lap));

% Constant RHS from Dirichlet BC (left wall T=1, right wall T=0)
rhs_bc = zeros(N, 1);
for iy = 1:n_y
    kk = (1-1)*n_y + iy;   % ix_int=1 (leftmost interior column)
    rhs_bc(kk) = (1/dh^2) * 1.0;   % left wall T=1 contribution
end

%% ── Solve per sample ─────────────────────────────────────────────────────────
fprintf('\nSolving %d FDM problems (per-sample factorisation) ...\n', n_total);
T_all = zeros(res, res, n_total);

for k = 1:n_total
    if mod(k, 200)==0 || k==1
        fprintf('  Sample %4d / %d\n', k, n_total);
    end

    % Build diagonal reaction matrix for this sample: diag(h at DOF nodes)
    h_vec = zeros(N, 1);
    for ix_int = 1:n_x
        ix = ix_int + 1;
        for iy = 1:n_y
            kk = (ix_int-1)*n_y + iy;
            h_vec(kk) = h_all(iy, ix, k);
        end
    end
    H_diag = spdiags(h_vec, 0, N, N);

    % Full system: (A_lap + H_diag) * T = rhs_bc
    A_full = A_lap + H_diag;

    % Solve (direct sparse)
    T_vec = A_full \ rhs_bc;

    % Reconstruct full res x res field
    T_full = zeros(res, res);
    T_full(:, 1)   = 1.0;   % left Dirichlet
    T_full(:, res) = 0.0;   % right Dirichlet
    for ix_int = 1:n_x
        ix = ix_int + 1;
        T_full(:, ix) = T_vec((ix_int-1)*n_y+1 : ix_int*n_y);
    end
    T_all(:,:,k) = T_full;
end

fprintf('  T range: [%.4f, %.4f]\n', min(T_all(:)), max(T_all(:)));

% Sanity checks
assert(max(abs(T_all(:,1,:)-1.0), [], 'all') < 1e-10, 'Left BC violated');
assert(max(abs(T_all(:,res,:)-0.0), [], 'all') < 1e-10, 'Right BC violated');
fprintf('  Dirichlet BCs: OK\n');

% Physical check: T should decrease from left to right for all samples
T_left_mean  = mean(T_all(:,1,:), 'all');
T_right_mean = mean(T_all(:,res,:), 'all');
fprintf('  Mean T at left=%.4f (expect 1.0), right=%.4f (expect 0.0)\n', ...
    T_left_mean, T_right_mean);

%% ── Split ─────────────────────────────────────────────────────────────────────
h_train     = h_all(:,:,1:n_train);
T_train     = T_all(:,:,1:n_train);
alpha_train = alpha_all(1:n_train);
beta_train  = beta_all(1:n_train);

h_test      = h_all(:,:,n_train+1:end);
T_test      = T_all(:,:,n_train+1:end);
alpha_test  = alpha_all(n_train+1:end);
beta_test   = beta_all(n_train+1:end);

%% ── Save ──────────────────────────────────────────────────────────────────────
% Saved as (n, res, res) -- Python loads directly without .T
% X, Y still use .T convention (2D grids)
X_store = X_mesh';
Y_store = Y_mesh';

% Training file
train_path = fullfile(out_dir, 'convdiff_train.mat');
if exist(train_path,'file'), delete(train_path); end
coeff   = h_train;    % already (res, res, n) -- h5py reads as (n, res, res)
sol_fdm = T_train;
X       = X_store;                      %#ok<NASGU>
Y       = Y_store;                      %#ok<NASGU>
alpha   = alpha_train;                  %#ok<NASGU>
beta    = beta_train;                   %#ok<NASGU>
save(train_path, 'coeff', 'sol_fdm', 'X', 'Y', 'alpha', 'beta', '-v7.3');
fprintf('\nSaved: %s\n', train_path);

% Test file
test_path = fullfile(out_dir, 'convdiff_test.mat');
if exist(test_path,'file'), delete(test_path); end
coeff   = permute(h_test, [3,1,2]);    %#ok<NASGU>
sol_fdm = permute(T_test, [3,1,2]);    %#ok<NASGU>
alpha   = alpha_test;                   %#ok<NASGU>
beta    = beta_test;                    %#ok<NASGU>
save(test_path, 'coeff', 'sol_fdm', 'X', 'Y', 'alpha', 'beta', '-v7.3');
fprintf('Saved: %s\n', test_path);

%% ── Verification plot ─────────────────────────────────────────────────────────
k_show = 1;
figure('Name','Verification','NumberTitle','off','Position',[100 100 900 350]);

subplot(1,2,1);
contourf(X_mesh, Y_mesh, h_test(:,:,k_show), 40, 'LineStyle','none');
colorbar; axis equal tight;
title(sprintf('h(x,y) = exp(%.2f sin(pi x) + %.2f cos(pi y))  [sample %d]', ...
    alpha_test(k_show), beta_test(k_show), k_show));
xlabel('x'); ylabel('y');

subplot(1,2,2);
contourf(X_mesh, Y_mesh, T_test(:,:,k_show), 40, 'LineStyle','none');
colorbar; axis equal tight;
title(sprintf('T(x,y) FDM solution  [sample %d]', k_show));
xlabel('x'); ylabel('y');

sgtitle('-Lap(T) + h(x,y)*T = 0  |  T(0,y)=1, T(1,y)=0, dT/dy=0 at y=0,1');
fprintf('\nDone.\n');
fprintf('  convdiff_train.mat -- %d samples\n', n_train);
fprintf('  convdiff_test.mat  -- %d samples\n', n_test);