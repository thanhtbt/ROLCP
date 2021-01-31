%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% written by NGUYEN Viet-Dung, Karim Abed-Meraim and Nguyen Linh-Trung
% modified from programs by D. Nion and N.D. Sidiropoulos
%
% If you find our program useful, please cite to: 
% Nguyen Viet-Dung, Karim Abed-Meraim and Nguyen Linh-Trung
% "Second-order optimization based adaptive PARAFAC decomposition of
% three-way tensors"
% Digital Signal processing, 63:100-111, April 2017
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% script to test PARAFAC tracking algorithm
% At instant t, matrices A(t) (IxR), C(t) (KxR) and B(t) (J0xR) are created.
% At instant t+1, matrices A(t+1) and C(t+1) are built from A(t) and C(t) with 
% an additive perturbation (the norm of the pertubation matrix corresponds to 
% the speed of variation), and B(t+1) is created by appending a new row to B(t).

clear 
% close all
clc

%******************************************
% PARAMETERS for the speed of variation
%******************************************
TT = 1000;  % Duration of the tracking procedure: TT new slices will be observed

%TT = 10^5; % comment out this line if you want to verify stability as stated in our paper, note
% that it will take time to run, depending on the performance of your
% computer



% Note: when A and B are varying we don't have a PARAFAC model anymore !
% The PARAFAC model is exactly valid at each tracking step if etaA and etaC=0,
% and if no noise is added, i.e., the TT new slices appended are always 
% generated with the same loadings.
% So a small value of etaA and etaC yields a slow degradation of the PARAFAC 
% model, i.e., the estimation error of the loading matrices will increase
% to a certain point during the tracking procedure.
% This estimation error is increasing quickly when the model degradation is fast
% i.e., if A and C are varying a lot between 2 tracking steps.

%********************************
% DIMENSIONS
%*******************************
I=10;
K=12;
R=10;  
J=20;                % initial value of the time varying dimension 
                     % (number of initially observed slices)
size_vec=[I J K R];  % vector of size
SNR=30; inf;             % To add noise or not on initial tensor. 
                     % Choose SNR=inf for a noise free model
data_type='real';    % 'real' to generate real data or 'complex' for complex data
var = 10^-3;      % noise variance to add to model
%****************************
% Algorithms parameters
%****************************
wind='exp';      % window: 'exp' for exponential or 'trunc' for truncated
L=0.9;             % forgetting factor 
N=30;              % length of window (for wind='trunc' only), N<=J
Niter=1;           % nb of iterations for optional inner loop of RLST
 
%---------------------------------------------
%-------  Create data at time 0 -------------
%---------------------------------------------
if strcmp(data_type,'real')==1
    A = randn(I,R);B=randn(J,R);C=randn(K,R);
elseif strcmp(data_type,'complex')==1
    A=randn(I,R)+1i*randn(I,R);B=randn(J,R)+1i*randn(J,R);C=randn(K,R)+1i*randn(K,R);
end
    
% Create observed tensor that follows PARAFAC model
X=zeros(I,J,K);
for k=1:K
    X(:,:,k)=A*diag(C(k,:))*B.';
end

% Add noise
if strcmp(data_type,'real')==1
    Noise_tens=randn(I,J,K);
elseif strcmp(data_type,'complex')==1
   Noise_tens=randn(I,J,K)+1i*randn(I,J,K);
end
sigma=(10^(-SNR/20))*(norm(reshape(X,J*I,K),'fro')/norm(reshape(Noise_tens,J*I,K),'fro'));
X=X+sigma*Noise_tens;

% KIxJ matrix unfolding
Xm = reshape(permute(X,[1 3 2]),K*I,J);


%------------------------------ INITIALIZATION Of TRACKING ------------------------------------------------
% Given the initially observed tensor X consisting of J slices, we need to get initial estimates of 
% the loading matrices. These estimates can be obtained by computing the PARAFAC decomposition of X 
% with a batch algorithm (e.g. ALS, ALS+Line Search, Levenberg-Marquardt, Simultaneous Diagonalization).
% One can also get initial estimates by using the Direct Trilinear Decomposition (DTLD) which, 
% for noise-free data, gives a closed form solution to the PARAFAC decomposition, provided that 
% 2 dimensions are greater than the rank R.
%----------------------------------------------------------------------------------------------------------

%% ---- Initialize PARAFAC-RLST with batch ALS algorithm  --------------
[A1, B1 ,C1]=cp3_alsls(X,R);
% A1 = randn(I,R);
% B1 = randn(J,R);
% C1 = randn(K,R);

switch lower(wind)
    case('exp')
        R1=Xm*conj(B1);
        P1=B1.'*conj(B1);
        Z1=pinv(R1);
        Q1=inv(P1);
    case('trunc')
        R1=Xm(:,end-N+1:end)*conj(B1(end-N+1:end,:));
        P1=B1(end-N+1:end,:).'*conj(B1(end-N+1:end,:));
        Z1=pinv(R1);
        Q1=inv(P1);
end

%% ---- Initialize SOAPARAFAC with batch ALS algorithm  --------------

A4 = A1;
C4 = C1;
B4 = B1;
H4 = kat_rao(C4,A4);
H4inv = pinv(H4);
Rinv4 = pinv(B4'*B4);

% %% Initialize Batch ALS that will be used repeatedly on the data (benchmark)
% Xw = Xm;
% A_ALS=A1;B_ALS=B1;C_ALS=C1;


%*******************************************************************************
%  START  TRACKING SIMULATION
%  Acquisition of Successive Observations
%  and comparison between RLST and Batch
%-------------------------------------------------------------------------------
% NOTE: an important feature of tracking algorithms is that we do not need to 
% store a tensor with the second dimension increasing. 
% Since the updates are recursive, only the most recent slice has to be stored, 
% and in case of a truncated window, the slice observed N steps ago.
% So there is no need to build Xm = [Xm,x] which has a growing dimension; this 
% avoids memory problems if TT (the number of slices added) is big.
% A matrix with growing dimensions is built in the code below only for the purpose
% of using the batch ALS algorithm, used as a benchmark (the PARAFAC decomposition
% is computed repeatedly on a tensor of growing size).
%********************************************************************************
NUMBLOOPS = 1;
for numbLoop = 1: NUMBLOOPS
    disp(['Number of loop: ', num2str(numbLoop)]);
    for t=1:TT   % TT tracking steps        
     
        %% --------- Time-varying Model for loading matrices --------------
        if t == 300 || t == 600
            etaA =  10^-1;
            etaC = etaA;
            disp(['Trial mode: ', num2str(t)]);
        else
            etaA =  10^-3;
            etaC = etaA;
        end
        if strcmp(data_type,'real')==1
            A = (1-etaA) * A + etaA * randn(I,R);
            C = (1-etaC) * C + etaC * randn(K,R);
            b = randn(1,R);       % new row of B in column format
        elseif strcmp(data_type,'complex')==1
            A = (1-etaA) * A + etaA * (randn(I,R)+1i*randn(I,R));
            C = (1-etaC) * C + etaC * (randn(K,R)+1i*randn(K,R));
            b = randn(1,R)+1i*randn(1,R);       % new row of B
        end
        x = kat_rao(C,A)*b.' + var*randn(I*K,1);              % new slice in vector format, Gaussian noise

        %% ------------FastSOAP -------------------------------------------
        [b4,A4,C4,H4,H4inv,Rinv4] = complexFastSOAParafac(x,A4,C4,H4,H4inv,Rinv4,I,K,R,t);
        B4 = [B4; b4];
%         MSE_A4(numbLoop,t) = subspace(A4,A);
%         MSE_C4(numbLoop,t) = subspace(C4,C);
        
        [~,~,~,MSE_A4(numbLoop,t)]=solve_perm_scale(A4,A);
        [~,~,~,MSE_C4(numbLoop,t)]=solve_perm_scale(C4,C);
        [~,~,~,MSE_x4(numbLoop,t)]=solve_perm_scale(kat_rao(C4,A4)*b4.',x);
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    end     %%%%%%% END of TRACKING SIMULATION  %%%%%%%%%%%%%%%%%
end
%% calculate average Monte Carlo run
aveMSE_A4 = (1/NUMBLOOPS)*sum(MSE_A4,1);

aveMSE_C4 = (1/NUMBLOOPS)*sum(MSE_C4,1);

aveMSE_x4 = (1/NUMBLOOPS)*sum(MSE_x4,1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% PLOT PERFORMANCE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

figure
semilogy(1:TT,aveMSE_A4,'-');hold on;
legend('SOAP');
grid on;
xlabel('time','interpreter','latex','FontSize',20);
ylabel('STD of $\mathbf{A}$','interpreter','latex','FontSize',20);
title('Evolution of STD of $\mathbf{A}$','interpreter','latex','FontSize',20);

% Plot evolution of MSE of x vs tracking steps
figure
semilogy(1:TT,aveMSE_x4,'-');hold on;
legend('SOAP');
grid on;
xlabel('time','interpreter','latex','FontSize',20);
ylabel('STD of $\mathbf{x}$','interpreter','latex','FontSize',20);
title('Evolution of STD of $\mathbf{x}$','interpreter','latex','FontSize',20);

% Plot evolution of MSE of C vs tracking steps
% figure
% semilogy(1:TT,aveMSE_C4,'-');hold on;
% legend('SOAP');
% grid on;
% xlabel('time','interpreter','latex','FontSize',20);
% ylabel('STD of $\mathbf{C}$','interpreter','latex','FontSize',20);
% title('Evolution of STD of $\mathbf{C}$','interpreter','latex','FontSize',20);

