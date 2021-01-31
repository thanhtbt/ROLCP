function [Factor,PER] = PARAFAC_SDT_Mod(X,R,OPTS)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% PARAFAC_SDT algorithm for tracking CP decomposition of 3-way tensors
% Authors  : Nion, D. & Sidiropoulos, N. D.
% Reference:  [1] Nion, D. & Sidiropoulos, N. D.
%             "Adaptive Algorithms to Track the PARAFAC Decomposition of a Third-Order Tensor" 
%             IEEE Trans. Signal Process., 2009, 57, 2299-2310.
% Edited    : Le Trung Thanh (31/1/2021)
% Contact   : letrungthanhtbt@gmail.com
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if nargin < 3
    flag = 0; % without performance estimation part
    Factors = [];
    PER     = [];
else
    flag = 1; % with performance estimation part
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
end

[I,J,T]    = size(X);

%% Performance Evaluation
PER.A  = zeros(1,T);
PER.B  = zeros(1,T);
PER.X  = zeros(1,T);

%% Algorithm's Parmaters
L=0.9;             % forgetting factor
N = J;              % length of window (for wind='trunc' only), N<=J
%% Initialization
t_train = max(I,J);
X_train = X(:,:,1:t_train);
Xm = reshape(permute(X_train,[1 2 3]),I*J,t_train);
Xw2 = Xm(:,end-N+1:end)* diag(L.^((N-(1:N))/2));
[U2,S2,V2]=svd(Xw2,0);
U2=U2(:,1:R);
S2=S2(1:R,1:R);
V2=V2(:,1:R);
E2=U2*S2;
Tw2=permute(reshape(Xw2,I,J,size(Xw2,2)),[1 2 3]);
[A2, B2 ,C2]=cp3_alsls(Tw2,R);
W2  = pinv(E2)*kat_rao(B2,A2);
Wi2 = inv(W2);
V2  = V2(end-N+1:end,:);

%% STREAMING
for t=1:T
    Xt = X(:,:,t);
    x = (Xt(:));
    
    [c2,A2,B2,W2,Wi2,V2,S2,U2]=parafac_sdt(x,A2,W2,Wi2,V2,S2,U2,L);
    C2 = [C2; c2.'];
    
    %% Performance Evaluation
    At = A2;
    Bt = B2;
    
     if flag == 1
        % Data
        if flag_slide == 1
            Xt_true    = Slide_true{1,t};
        else
            Xt_true = Xt;
        end
        Ht      = khatrirao(Bt,At);
        ct      = Ht \ Xt(:);
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
end
