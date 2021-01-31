function [Factor,PER] = ROLCP(X,R,OPTS)
%% Randomized Online CP Algorithm
% Author     : LE Trung Thanh
% Contact    : letrungthanhtbt@gmail.com  
% Reference  : L.T. Thanh, K. Abed-Meraim, N.L. Trung and A.Hafiane
%              "A fast randomized adaptive CP decomposition for streaming tensors".
%              IEEE-ICASSP, 2021. 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[I,J,T] = size(X); 
if nargin < 3  flag = 0; % without performance estimation part
    Factors = [];
    PER     = [];
    
else flag = 1; % with performance estimation part
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
    % Performance Evaluation
    PER.A  = zeros(1,T);
    PER.B  = zeros(1,T);
    PER.X  = zeros(1,T);
end

%% Random Initialization
At = randn(I,R);
Bt = randn(J,R);
Ct = [];
m  = round(10*R * log(I*J));

for t   =  1 : T
    Xt       = X(:,:,t);
    Omega_t  = ones(I,J); 
    %% Estimate ct
    Ht       = khatrirao(Bt,At);
    xt       = (Xt(:));
    idx      = find(xt);
    Ht_Omega = Ht(idx,:);
    xt_Omega = xt(idx,:);   
    if length(idx) > m
        idx_Sam  = randsample(length(idx),m);
    else
        idx_Sam = [1:length(idx)];
    end
    ct       = Ht_Omega(idx_Sam,:) \ xt_Omega(idx_Sam,:);
    Ct       = [Ct; ct'];
 
    %% Estimate Factors
    At = updateFactor(At,Xt,ct,Bt,Omega_t);
    Bt = updateFactor(Bt,Xt',ct,At,Omega_t');
   
    %% Performance Evaluation
    if flag == 1  
        % Data
        if flag_slide == 1
            Xt_true    = Slide_true{1,t};
        else
            Xt_true = Xt;
        end
            Ht      = khatrirao(Bt,At);
            xt_re   = Ht * ct;
            xt_true = (Xt_true(:));
            er      = (xt_re - xt_true);
            PER.X(1,t)  = PER.X(1,t)  + norm(er)/norm(xt_true);
       
        % Factors 
        if flag_factor == 1
            Factors_t   = Factors{1,t};
            A_true      = Factors_t{1,1};
            B_true      = Factors_t{1,2};
            [~,~,~,erA] = solve_perm_scale(At,A_true);
            [~,~,~,erB] = solve_perm_scale(Bt,B_true);
            PER.A(1,t)  = PER.A(1,t)  + erA/norm(A_true);
            PER.B(1,t)  = PER.B(1,t)  + erB/norm(B_true);
        else
        end
    end
    
    
 
    
end
Factor.A = At;
Factor.B = Bt;
Factor.C = Ct;

end

function  U = updateFactor(U_old,X,lambda_t,V_t,Omega_t)

[I,r]  = size(U_old);
J      = size(V_t,1);

W  =  (V_t * diag(lambda_t))';
H  =   W * W' + 10*eye(r);
S_inv    = H \ W;
X_re     = Omega_t.*(U_old * W);
Residual = Omega_t.*(X - X_re);
DeltaU   = Residual * S_inv';
U        = U_old  +  DeltaU;

end


