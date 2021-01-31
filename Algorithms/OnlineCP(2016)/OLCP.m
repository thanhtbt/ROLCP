function [Factor,PER] = OLCP(X,R,OPTS)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% OLCP algorithm for tracking CP decomposition of 3-way tensors
% Created  : Le Trung Thanh 
% Contact  : letrungthanhtbt@gmail.com  

% Reference:  [1] Zhou, S.; Vinh, N. X.; Bailey, J.; Jia, Y. & Davidson, I.
%                "Accelerating online CP decompositions for higher order tensors"
%                 ACM Int. Conf. Knowl. Discover. Data Min., 2016, 1375-1384.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Input

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


tensor_dims = size(X);
I = tensor_dims(1);
J = tensor_dims(2);
T = tensor_dims(3);


%% Performance Evaluation
PER.A  = zeros(1,T);
PER.B  = zeros(1,T);
PER.X  = zeros(1,T);

%% Good Initialization
t_train = 100;
X_train = X(:,:,1:t_train);

[A1, B1 ,C1]=cp3_alsls(X_train,R);

H1 = khatrirao(B1,A1);
At = A1;
Bt = B1;
Ct = C1;
Ht = H1;

X_train_m1 = ten2mat(tensor(X_train),1);
CBt = khatrirao(Ct,Bt);
Pt  = X_train_m1 * CBt;
Qt  = CBt' * CBt;
X_train_m2 = ten2mat(tensor(X_train),2);
CAt = khatrirao(Ct,At);
Ut  = X_train_m2 * CAt;
Vt  = CAt' * CAt;

for t   =  1 : T
    
    Xt  = X(:,:,t);  xt  = (Xt(:));
    %% Estimate Ct
    Ht = khatrirao(Bt,At);
    ct = Ht \ xt;
    Ct = [Ct; ct'];
    %% Estimate At,Bt
    P  = Xt * khatrirao(ct',Bt);
    Pt = Pt + P;
    Q  = (ct * ct') .* (Bt'*Bt);
    Qt = Qt + Q;
    At = Pt * inv(Qt);
    
    U  = Xt' * khatrirao(ct',At);
    Ut = Ut + U;
    V  = (ct * ct') .* (At'*At);
    Vt = Vt + V;
    Bt = Ut * inv(Vt);
    
     
    %%  Performance Analysis
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

