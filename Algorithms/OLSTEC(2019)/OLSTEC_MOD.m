function [Xsol,sub_infos,PER] = OLSTEC_MOD(A_in, OPTS, options);
%% OLSTEC algorithm
% Author   : H.Kasai 
% Reference: H.Kasai,"Fast online low-rank tensor subspace tracking by CP decomposition using recursive least squares from incomplete observations"
%            Neurocomputing, 2019.

%%

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



% extract options
if ~isfield(options, 'maxepochs')
    maxepochs = 1;
else
    maxepochs = options.maxepochs;
end

if ~isfield(options, 'tolcost')
    tolcost = 1e-12;
else
    tolcost = options.tolcost;
end

if ~isfield(options, 'permute_on')
    permute_on = false;
else
    permute_on = options.permute_on;
end

if ~isfield(options, 'lambda')
    lambda = 0.7;
else
    lambda = options.lambda;
end

if ~isfield(options, 'mu')
    mu = 0.1;
else
    mu = options.mu;
end

if ~isfield(options, 'tw_flag')
    TW_Flag = false;
else
    TW_Flag = options.tw_flag;
end

if ~isfield(options, 'tw_len')
    TW_LEN = 10;
else
    TW_LEN = options.tw_len;
end

if ~isfield(options, 'store_subinfo')
    store_subinfo = true;
else
    store_subinfo = options.store_subinfo;
end

if ~isfield(options, 'store_matrix')
    store_matrix = false;
else
    store_matrix = options.store_matrix;
end

if ~isfield(options, 'verbose')
    verbose = 2;
else
    verbose = options.verbose;
end




% set tensor dimentions
rows            = tensor_dims(1);
cols            = tensor_dims(2);
slice_length    = tensor_dims(3);


% initialize X (A_t0 and B_t0) if needed

A               = A_in;             % Full entries
A_Omega         = Omega.*A_in;

A_t0 = Xinit.A;
B_t0 = Xinit.B;
C_t0 = Xinit.C;


% prepare Rinv histroy buffers
RAinv = repmat(100*eye(Rank), rows, 1);
RBinv = repmat(100*eye(Rank), cols, 1);

% prepare
N_AlphaAlphaT = zeros(Rank*rows, Rank*(TW_LEN+1));
N_BetaBetaT   = zeros(Rank*cols, Rank*(TW_LEN+1));

% prepare
N_AlphaResi = zeros(Rank*rows, TW_LEN+1);
N_BetaResi  = zeros(Rank*cols, TW_LEN+1);


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
if isfield(OPTS,'TrueFactors')
    Factors  = OPTS.TrueFactors;
    flag_factor = 1;
else
    flag_factor = 0;
end
if isfield(OPTS,'TrueSlides')
    Slide_true = OPTS.TrueSlides;
    flag_slide = 1;
else
    flag_slide = 0;
end



PER.A  = zeros(1,slice_length);
PER.B  = zeros(1,slice_length);
PER.X  = zeros(1,slice_length);

% if verbose > 1
%     fprintf('OLSTEC Epoch 000, Cost %7.3e, Cost(test) %7.3e\n', train_cost, test_cost);
% end


% Main loop
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
        
        % Pull out the relevant indices and revealed entries for this column
        % sampled original image
        I_mat = A(:,:, col_order(k));
        Omega_mat   = Omega(:,:, col_order(k));
        I_mat_Omega = A_Omega(:,:, col_order(k));
        
        
        %% Gamma (B) Update
        temp3 = 0;
        temp4 = 0;
        for m=1:rows
            alpha_remat = repmat(A_t0(m,:)', 1, cols);
            alpha_beta = alpha_remat .* B_t0';
            I_row = I_mat_Omega(m,:);
            temp3 = temp3 + alpha_beta * I_row';
            
            Omega_mat_ind = find(Omega_mat(m,:));
            alpha_beta_Omega = alpha_beta(:,Omega_mat_ind);
            temp4 = temp4 + alpha_beta_Omega * (alpha_beta_Omega');
        end
        
        temp4 = lambda * eye(Rank) + temp4;
        gamma = temp4 \ temp3;                                             % equation (18)
        
        
        %% update A
        for m=1:rows
            
            Omega_mat_ind = find(Omega_mat(m,:));
            I_row = I_mat_Omega(m,:);
            I_row_Omega = I_row(Omega_mat_ind);
            C_t0_Omega = B_t0(Omega_mat_ind,:);
            N_alpha_Omega = diag(gamma) * C_t0_Omega';
            N_alpha_alpha_t_Omega = N_alpha_Omega * N_alpha_Omega';
            
            % Calc TAinv (i.e. RAinv)
            TAinv = lambda^(-1) * RAinv((m-1)*Rank+1:m*Rank,:);
            if TW_Flag
                Oldest_alpha_alpha_t = N_AlphaAlphaT((m-1)*Rank+1:m*Rank,1:Rank);
                TAinv = inv(inv(TAinv) + N_alpha_alpha_t_Omega + (mu - lambda*mu)*eye(Rank) - lambda^TW_LEN * Oldest_alpha_alpha_t);
            else
                TAinv = inv(inv(TAinv) + N_alpha_alpha_t_Omega + (mu - lambda*mu)*eye(Rank));
            end
            
            % Calc delta A_t0(m,:)
            recX_col_Omega = N_alpha_Omega' * A_t0(m,:)';
            resi_col_Omega = I_row_Omega' - recX_col_Omega;
            N_alpha_Resi_Omega = N_alpha_Omega * diag(resi_col_Omega);
            
            N_resi_Rt_alpha = TAinv * N_alpha_Resi_Omega;
            delta_A_t0_m = sum(N_resi_Rt_alpha,2);
            
            % Update A
            if TW_Flag
                % update A
                Oldest_alpha_resi = N_AlphaResi((m-1)*Rank+1:m*Rank,1)';
                %A_t1(m,:) = A_t0(m,:) + delta_A_t0_m' - lambda^TW_LEN * Oldest_alpha_resi;
                A_t1(m,:) = A_t0(m,:)  - (mu - lambda*mu) * A_t0(m,:) * TAinv' + delta_A_t0_m' - lambda^TW_LEN * Oldest_alpha_resi;
                
                % Store data
                N_AlphaAlphaT((m-1)*Rank+1:m*Rank,TW_LEN*Rank+1:(TW_LEN+1)*Rank) = N_alpha_alpha_t_Omega;
                N_AlphaResi((m-1)*Rank+1:m*Rank,TW_LEN+1) = sum(N_alpha_Resi_Omega,2);
            else
                %A_t1(m,:) = A_t0(m,:) + delta_A_t0_m';
                %A_t1(m,:) = A_t0(m,:) - (mu - lambda*mu) * (TAinv * A_t0(m,:)')' + delta_A_t0_m';
                A_t1(m,:) = A_t0(m,:) - (mu - lambda*mu) * A_t0(m,:) * TAinv' + delta_A_t0_m';
            end
            
            % Store RAinv
            RAinv((m-1)*Rank+1:m*Rank,:) = TAinv;
        end
        
        % Final update of A
        A_t0 = A_t1;
        
        
        %% update B
        for n=1:cols
            
            Omega_mat_ind = find(Omega_mat(:,n));
            I_col = I_mat_Omega(:,n);
            I_col_Omega = I_col(Omega_mat_ind);
            A_t0_Omega = A_t0(Omega_mat_ind,:);
            N_beta_Omega = A_t0_Omega * diag(gamma);
            N_beta_beta_t_Omega = N_beta_Omega' * N_beta_Omega;
            
            % Calc TBinv (i.e. RBinv)
            TBinv = lambda^(-1) * RBinv((n-1)*Rank+1:n*Rank,:);
            if TW_Flag
                Oldest_beta_beta_t = N_BetaBetaT((n-1)*Rank+1:n*Rank,1:Rank);
                TBinv = inv(inv(TBinv) + N_beta_beta_t_Omega + (mu - lambda*mu)*eye(Rank) - lambda^TW_LEN * Oldest_beta_beta_t);
            else
                TBinv = inv(inv(TBinv) + N_beta_beta_t_Omega + (mu - lambda*mu)*eye(Rank));
            end
            
            % Calc delta B_t0(n,:)
            recX_col_Omega = B_t0(n,:) * N_beta_Omega';
            resi_col_Omega = I_col_Omega' - recX_col_Omega;
            N_beta_Resi_Omega = N_beta_Omega' * diag(resi_col_Omega);
            N_resi_Rt_beta = TBinv * N_beta_Resi_Omega;
            delta_C_t0_n = sum(N_resi_Rt_beta,2);
            
            if TW_Flag
                % Upddate B
                Oldest_beta_resi = N_BetaResi((n-1)*Rank+1:n*Rank,1)';
                B_t1(n,:) = B_t0(n,:) - (mu - lambda*mu) * B_t0(m,:) * TBinv' + delta_C_t0_n' - lambda^TW_LEN * Oldest_beta_resi;
                
                % Store data
                N_BetaBetaT((n-1)*Rank+1:n*Rank,TW_LEN*Rank+1:(TW_LEN+1)*Rank) = N_beta_beta_t_Omega;
                N_BetaResi((n-1)*Rank+1:n*Rank,TW_LEN+1) = sum(N_beta_Resi_Omega,2);
            else
                B_t1(n,:) = B_t0(n,:) - (mu - lambda*mu) * B_t0(n,:) * TBinv' + delta_C_t0_n';
                
            end
            
            % Store RBinv
            RBinv((n-1)*Rank+1:n*Rank,:) = TBinv;
        end
        
        
        if TW_Flag
            N_AlphaAlphaT(:,1:Rank) = [];
            N_BetaBetaT(:,1:Rank) = [];
            N_AlphaResi(:,1) = [];
            N_BetaResi(:,1) = [];
        end
        
        % Final update of B
        B_t0 = B_t1;
        
        
        %% Reculculate gamma (B)
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
        gamma = temp4 \ temp3;
        
        % Store gamma into C_t0
        C_t0(col_order(k),:) = gamma';
        
        % Reconstruct Low-Rank Matrix
        L_rec = A_t0 * diag(gamma) * B_t0';
        
        %% Performance Estimation
        if flag_slide == 1
            Xt_true  = Slide_true{1,k};
        else
            Xt_true = A(:,:, col_order(k));
        end
        if flag_factor == 1
            Factors_t  = Factors{1,k};
            A_true     = Factors_t{1,1};
            B_true     = Factors_t{1,2};
            [~,~,~,erA] = solve_perm_scale(A_t0,A_true);
            [~,~,~,erB] = solve_perm_scale(B_t0,B_true);
            PER.A(1,k)  = PER.A(1,k)  + erA/norm(A_true);
            PER.B(1,k)  = PER.B(1,k)  + erB/norm(B_true);
        else
            
        end
              
        er = L_rec - Xt_true;
        PER.X(1,k)  = PER.X(1,k)  + norm(er)/norm(Xt_true);
        
        %% 
        if store_subinfo
%             % Residual Error
%             norm_residual   = norm(I_mat(:) - L_rec(:));
%             norm_I          = norm(I_mat(:));
%             error           = norm_residual/norm_I;
%             sub_infos.inner_iter    = [sub_infos.inner_iter (outiter-1)*slice_length+k];
%             sub_infos.err_residual    = [sub_infos.err_residual error];
%             
%             % Running-average Estimation Error
%             if k == 1
%                 run_error   = error;
%             else
%                 run_error   = (sub_infos.err_run_ave(end) * (k-1) + error)/k;
%             end
%             sub_infos.err_run_ave     = [sub_infos.err_run_ave run_error];
%             
%             % Store reconstruction error
%             if store_matrix
%                 E_rec = I_mat - L_rec;
%                 sub_infos.I(:,k) = I_mat_Omega(:);
%                 sub_infos.L(:,k) = L_rec(:);
%                 sub_infos.E(:,k) = E_rec(:);
%             end
%             
%             for f=1:slice_length
%                 gamma = C_t0(f,:)';
%                 Rec(:,:,f) = A_t0 * diag(gamma) * B_t0';
%             end
%             
%             % Global train_cost computation
%             train_cost = compute_cost_tensor(Rec, Omega, A_Omega, tensor_dims);
%             if ~isempty(Gamma) && ~isempty(A_Gamma)
%                 test_cost = compute_cost_tensor(Rec, Gamma, A_Gamma, tensor_dims);
%             else
%                 test_cost = 0;
%             end
%             sub_infos.global_train_cost  = [sub_infos.global_train_cost train_cost];
%             sub_infos.global_test_cost  = [sub_infos.global_test_cost test_cost];
            
            
%             if verbose > 2
%                 fnum = (outiter-1)*slice_length + k;
%                 % fprintf('OLSTEC: fnum = %03d, cost = %e, error = %e\n', fnum, train_cost, error);
%             end
        end
        
    end
    
    
    % store infos
%     infos.iter = [infos.iter; outiter];
%     infos.time = [infos.time; infos.time(end) + toc(t_begin)];
    
%     if ~store_subinfo
%         for f=1:slice_length
%             gamma = C_t0(f,:)';
%             Rec(:,:,f) = A_t0 * diag(gamma) * B_t0';
%         end
%         
%         train_cost = compute_cost_tensor(Rec, Omega, A_Omega, tensor_dims);
%         if ~isempty(Gamma) && ~isempty(A_Gamma)
%             test_cost = compute_cost_tensor(Rec, Gamma, A_Gamma, tensor_dims);
%         else
%             test_cost = 0;
%         end
%     end
%     infos.train_cost = [infos.train_cost; train_cost];
%     infos.test_cost = [infos.test_cost; test_cost];
    
%     if verbose > 1
%         fprintf('OLSTEC Epoch %0.3d, Cost %7.3e, Cost(test) %7.3e\n', outiter, train_cost, test_cost);
%     end
    
%     % stopping criteria: cost tolerance reached
%     if train_cost < tolcost
%         fprintf('train_cost sufficiently decreased.\n');
%         break;
%     end
end

Xsol.A = A_t0;
Xsol.B = B_t0;
Xsol.C = C_t0;
end


