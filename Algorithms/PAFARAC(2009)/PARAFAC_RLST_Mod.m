function [Factor,PER] = PARAFAC_RLST_Mod(X,R,OPTS)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% PARAFAC_RLST algorithm for tracking CP decomposition of 3-way tensors
% Authors   : Nion, D. & Sidiropoulos, N. D.
% Reference : [1] Nion, D. & Sidiropoulos, N. D.
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
wind='exp';        % window: 'exp' for exponential or 'trunc' for truncated
L=0.9;             % forgetting factor
N=J;               % length of window (for wind='trunc' only), N<=J
Niter=1;           % nb of iterations for optional inner loop of RLST
%% Initialization
t_train = max(I,J);
X_train = X(:,:,1:t_train);
Xm = reshape(permute(X_train,[1 2 3]),I*J,t_train);
[A1, B1 ,C1]=cp3_alsls(X_train,R);
switch lower(wind)
    case('exp')
        R1=Xm*conj(C1);
        P1=C1.'*conj(C1);
        Z1=pinv(R1);
        Q1=inv(P1);
    case('trunc')
        R1=Xm(:,end-N+1:end)*conj(C1(end-N+1:end,:));
        P1=C1(end-N+1:end,:).'*conj(C1(end-N+1:end,:));
        Z1=pinv(R1);
        Q1=inv(P1);
end

%% STREAMING
for t=1:T
    Xt = X(:,:,t);
    x  = (Xt(:));
    
    xu = Xm(:,end-N+1);
    Xm = [Xm(:,2:end),x];
    bu = C1(end-N+1,:).';
    tic;
    [c1,A1,B1,P1,Q1,R1,Z1]=parafac_rlst(x,xu,bu,A1,P1,Q1,R1,Z1,wind,L,N,Niter);
    time(t,1)=toc;
    C1 = [C1(2:end,:); c1.'];
    
    %% Performance Evaluation
    At = A1;
    Bt = B1;
    
    
    if flag == 1
        % Data
        if flag_slide == 1
            Xt_true    = Slide_true{1,t};
        else
            Xt_true = Xt;
        end
        Ht      = khatrirao(Bt,At);
        ct      = Ht \ (Xt(:));
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
