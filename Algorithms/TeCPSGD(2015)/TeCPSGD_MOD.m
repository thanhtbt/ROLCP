function [Xsol, sub_infos, PER] = TeCPSGD(A_in, OPTS, options)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% TeSPSGD algorithm for  tracking CP decomposition of 3-way tensors.
% Reference: [1] M. Mardani, G. Mateos, and G.B. Giannakis,
%                "Subspace learning and imputation for streaming big data matrices and tensors,"
%                IEEE Trans. Signal Process., vol. 63, no. 10, pp. 266-2677, 2015.
% Created by H.Kasai on June 07, 2017

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if isfield(OPTS,'Omega_in')
    Omega_in = OPTS.Omega_in;
    Omega    = OPTS.Omega_in;
else
end
if isfield(OPTS,'Gamma_in')
    Gamma_in = OPTS.Gamma_in;
    Gamma    = Gamma_in;
    A_Gamma  = Gamma_in.*A_in;
else
    A_Gamma  = [];
    Gamma    = [];
end

if isfield(OPTS,'tensor_dims')
    tensor_dims = OPTS.tensor_dims;
else
    tensor_dims = size(A_in);
end

if isfield(OPTS,'Rank')
    Rank = OPTS.Rank;
else
end

if isfield(OPTS,'Xinit')
    Xinit      = OPTS.Xinit;
else
    Xinit.A = randn(tensor_dims(1), Rank);
    Xinit.B = randn(tensor_dims(2), Rank);
    Xinit.C = randn(tensor_dims(3), Rank);
end


A               = A_in;             % Full entries
A_Omega         = Omega_in.*A_in;   % Training entries i.e., Omega_in.*A_in

A_t0 = Xinit.A;
B_t0 = Xinit.B;
C_t0 = Xinit.C;

% set tensor size
rows            = tensor_dims(1);
cols            = tensor_dims(2);
slice_length    = tensor_dims(3);


% set options
lambda          = options.lambda;
mu              = options.mu;
stepsize_init   = options.stepsize;
maxepochs       = options.maxepochs;
tolcost         = options.tolcost;
store_subinfo   = options.store_subinfo;
store_matrix    = options.store_matrix;
verbose         = options.verbose;

if ~isfield(options, 'permute_on')
    permute_on = 1;
else
    permute_on = options.permute_on;
end


% calculate initial cost
Rec = zeros(rows, cols, slice_length);
for k=1:slice_length
    gamma = C_t0(k,:)';
    Rec(:,:,k) = A_t0 * diag(gamma) * B_t0';
end
train_cost = compute_cost_tensor(Rec, Omega, A_Omega, tensor_dims);
if ~isempty(Gamma) && ~isempty(A_Gamma)
    test_cost = compute_cost_tensor(Rec, Gamma, A_Gamma, tensor_dims);
else
    test_cost = 0;
end


% initialize infos
infos.iter = 0;
infos.train_cost = train_cost;
infos.test_cost = test_cost;
infos.time = 0;

% initialize sub_infos
sub_infos.inner_iter = 0;
sub_infos.err_residual = 0;
sub_infos.err_run_ave = 0;
sub_infos.global_train_cost = 0;
sub_infos.global_test_cost = 0;
if store_matrix
    sub_infos.I = zeros(rows * cols, slice_length);
    sub_infos.L = zeros(rows * cols, slice_length);
    sub_infos.E = zeros(rows * cols, slice_length);
end

%% 
Factors    = OPTS.TrueFactors;
Slide_true = OPTS.TrueSlides;

PER.A  = zeros(1,slice_length);
PER.B  = zeros(1,slice_length);
PER.X  = zeros(1,slice_length);
%% 
% set parameters
eta = 0;

% if verbose > 0
%     fprintf('TeCPSGD [%d] Epoch 000, Cost %7.3e, Cost(test) %7.3e, Stepsize %7.3e\n', stepsize_init, train_cost, test_cost, eta);
% end


% main loop
for outiter = 1 : maxepochs
    
    % permute samples
    if permute_on
        col_order = randperm(slice_length);
    else
        col_order = 1:slice_length;
    end
    
    % Begin the time counter for the epoch
    t_begin = tic();
    
    for k=1:slice_length
        
        fnum = (outiter - 1) * slice_length + k;
        
        % sampled original image
        I_mat = A(:,:,col_order(k));
        Omega_mat = Omega(:,:,col_order(k));
        I_mat_Omega = Omega_mat .* I_mat;
        
        % Reculculate gamma (C)
        temp3 = 0;
        temp4 = 0;
        for m=1:rows
            alpha_remat = repmat(A_t0(m,:)', 1, cols);
            alpha_beta = alpha_remat .* B_t0';
            I_row = I_mat_Omega(m,:);
            temp3 = temp3 + alpha_beta * I_row';
            
            Omega_mat_ind = find(Omega_mat(m,:));
            alpha_beta_Omega = alpha_beta(:,Omega_mat_ind);
            temp4 = temp4 + alpha_beta_Omega * alpha_beta_Omega';
        end
        temp4 = lambda * eye(Rank) + temp4;
        gamma = temp4 \ temp3;                                             % equation (18)
        
        L_rec = A_t0 * diag(gamma) * B_t0';
        diff = Omega_mat.*(I_mat - L_rec);
        
        if 0
            eta = 1/mu;
            %A_t1 = (1 - lambda/(fnum*mu)) * A_t0 + 1/mu * diff	* B_t0 * diag(gamma);   % equation (20)&(21)
            %B_t1 = (1 - lambda/(fnum*mu)) * B_t0 + 1/mu * diff' * A_t0 * diag(gamma);  % equation (20)&(22)
            A_t1 = (1 - lambda*eta/fnum) * A_t0 + eta * dif   * B_t0 * diag(gamma);   % equation (20)&(21)
            B_t1 = (1 - lambda*eta/fnum) * B_t0 + eta * diff' * A_t0 * diag(gamma);  % equation (20)&(22)
        else
            eta = stepsize_init/(1+lambda*stepsize_init*fnum);
            A_t1 = (1 - lambda*eta) * A_t0 + eta * diff *  B_t0 * diag(gamma);   % equation (20)&(21)
            B_t1 = (1 - lambda*eta) * B_t0 + eta * diff' * A_t0 * diag(gamma);  % equation (20)&(22)
        end
        
        % Reculculate weights
        %weights = pinv(A_t1) * I_mat_Omega * pinv(B_t1');
        %t = diag(weights);
        
        % Update of A and B
        A_t0 = A_t1;
        B_t0 = B_t1;
        
        % Reculculate gamma (C)
        temp3 = 0;
        temp4 = 0;
        for m=1:rows
            alpha_remat = repmat(A_t0(m,:)', 1, cols);
            alpha_beta = alpha_remat .* B_t0';
            I_row = I_mat_Omega(m,:);
            temp3 = temp3 + alpha_beta * I_row';
            
            Omega_mat_ind = find(Omega_mat(m,:));
            alpha_beta_Omega = alpha_beta(:,Omega_mat_ind);
            temp4 = temp4 + alpha_beta_Omega * alpha_beta_Omega';
        end
        temp4 = lambda * eye(Rank) + temp4;
        gamma = temp4 \ temp3;                                             % equation (18)
        
        % Store gamma into C_t0
        C_t0(col_order(k),:) = gamma';
        
        % Reconstruct Low-Rank Matrix
        L_rec = A_t0 * diag(gamma) * B_t0';
        %             if disp_flag
        %                 L{alg_idx} = [L{alg_idx} L_rec(:)];
        %             end
        
        %% Performance Estimation
        
        Xt_true    = Slide_true{1,k};
        Factors_t  = Factors{1,k};
        A_true     = Factors_t{1,1};
        B_true     = Factors_t{1,2};
        
        
        [~,~,~,erA] = solve_perm_scale(A_t0,A_true);
        [~,~,~,erB] = solve_perm_scale(B_t0,B_true);
        er = L_rec - Xt_true;
        
        PER.A(1,k)  = PER.A(1,k)  + erA/norm(A_true);
        PER.B(1,k)  = PER.B(1,k)  + erB/norm(B_true);
        PER.X(1,k)  = PER.X(1,k)  + norm(er)/norm(Xt_true);
        
        
        
        if store_matrix
            E_rec = I_mat - L_rec;
            %sub_infos.E = [sub_infos.E E_rec(:)];
            sub_infos.I(:,k) = I_mat_Omega(:);
            sub_infos.L(:,k) = L_rec(:);
            sub_infos.E(:,k) = E_rec(:);
        end
        
      end
    
    
    % store infos
    infos.iter = [infos.iter; outiter];
    infos.time = [infos.time; infos.time(end) + toc(t_begin)];
    
    if ~store_subinfo
        for f=1:slice_length
            gamma = C_t0(f,:)';
            Rec(:,:,f) = A_t0 * diag(gamma) * B_t0';
        end
        
        train_cost = compute_cost_tensor(Rec, Omega, A_Omega, tensor_dims);
        if ~isempty(Gamma) && ~isempty(A_Gamma)
            test_cost = compute_cost_tensor(Rec, Gamma, A_Gamma, tensor_dims);
        else
            test_cost = 0;
        end
    end
    infos.train_cost = [infos.train_cost; train_cost];
    infos.test_cost = [infos.test_cost; test_cost];
    
    if verbose > 0
        % fprintf('TeCPSGD [%d] Epoch %0.3d, Cost %7.3e, Cost(test) %7.3e, Stepsize %7.3e\n', stepsize_init, outiter, train_cost, test_cost, eta);
    end
    
    % stopping criteria: cost tolerance reached
    if train_cost < tolcost
        % fprintf('train_cost sufficiently decreased.\n');
        break;
    end
end

Xsol.A = A_t0;
Xsol.B = B_t0;
Xsol.C = C_t0;
end




