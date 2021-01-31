% script to test PARAFAC tracking algorithm
% At instant t, matrices A(t) (IxR), C(t) (KxR) and B(t) (J0xR) are created.
% At instant t+1, matrices A(t+1) and C(t+1) are built from A(t) and C(t) with 
% an additive perturbation (the norm of the pertubation matrix corresponds to 
% the speed of variation), and B(t+1) is created by appending a new row to B(t).

clear all
close all
clc

%******************************************
% PARAMETERS for the speed of variation
%******************************************
TT=200;        % Duration of the tracking procedure: TT new slices will be observed
etaA =1e-1;    % To control speed of variation for A between two observations 
etaC =1e-1;    % To control speed of variation for C between two observations 

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
I=20;
K=20;
R=8;  
J=50;                % initial value of the time varying dimension 
                     % (number of initially observed slices)
size_vec=[I J K R];  % vector of size
SNR=inf;             % To add noise or not on initial tensor. 
                     % Choose SNR=inf for a noise free model
data_type='complex';    % 'real' to generate real data or 'complex' for complex data

%****************************
% Algorithms parameters
%****************************
wind='trunc';      % window: 'exp' for exponential or 'trunc' for truncated
L=0.9;             % forgetting factor 
N=10;              % length of window (for wind='trunc' only), N<=J
Niter=1;           % nb of iterations for optional inner loop of RLST
 
%---------------------------------------------
%-------  Create data at time 0 -------------
%---------------------------------------------
if strcmp(data_type,'real')==1
    A=randn(I,R);B=randn(J,R);C=randn(K,R);
elseif strcmp(data_type,'complex')==1
    A=randn(I,R)+j*randn(I,R);B=randn(J,R)+j*randn(J,R);C=randn(K,R)+j*randn(K,R);
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
   Noise_tens=randn(I,J,K)+j*randn(I,J,K);
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

%---- Initialize PARAFAC-RLST with batch ALS algorithm  --------------
[A1, B1 ,C1]=cp3_alsls(X,R);
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

%---- Initialize PARAFAC-SDT with batch ALS algorithm  --------------
% Use outputs from ALS to generate matrices useful to initialize PARAFAC-SDT
Xw2 = Xm(:,end-N+1:end)* diag(L.^((N-(1:N))/2));
[U2,S2,V2]=svd(Xw2,0);
U2=U2(:,1:R);
S2=S2(1:R,1:R);
V2=V2(:,1:R);
E2=U2*S2;
Tw2=permute(reshape(Xw2,I,K,size(Xw2,2)),[1 3 2]);
[A2, B2 ,C2]=cp3_alsls(Tw2,R);
W2 = pinv(E2)*kat_rao(C2,A2);
Wi2 = inv(W2);
V2=V2(end-N+1:end,:);

% ----- Initialize Batch ALS that will be used repeatedly on the data (benchmark)-------
Xw = Xm;
A_ALS=A1;B_ALS=B1;C_ALS=C1;


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
for t=1:TT   % TT tracking steps

    clc
    disp(['Tracking step ',num2str(t),' out of ',num2str(TT)])
    %--------------------------------------------------
    % Acquisition of a new set of KI samples and append
    % new slice along second dimension
    % Use time varying models for loading matrices
    % (not exploited by algorithms).
    %--------------------------------------------------
    
    %--------- Time-varying Model for loading matrices -----------------------------------
    if strcmp(data_type,'real')==1
        A = (1-etaA) * A + etaA * randn(I,R);
        C = (1-etaC) * C + etaC * randn(K,R);          
        b = randn(1,R);       % new row of B in column format
    elseif strcmp(data_type,'complex')==1
        A = (1-etaA) * A + etaA * (randn(I,R)+j*randn(I,R));
        C = (1-etaC) * C + etaC * (randn(K,R)+j*randn(K,R));          
        b = randn(1,R)+j*randn(1,R);       % new row of B 
    end
    x = kat_rao(C,A)*b.';              % new slice in vector format
  %------------  PARAFAC_SDT ----------------------------------
  tic;
  [b2,A2,C2,W2,Wi2,V2,S2,U2]=parafac_sdt(x,A2,W2,Wi2,V2,S2,U2,L);
  time(t,2)=toc;
  B2 = [B2; b2.'];
  [tmp,tmp,tmp,MSE_A(t,2)]=solve_perm_scale(A2,A);
  [tmp,tmp,tmp,MSE_C(t,2)]=solve_perm_scale(C2,C);
  [tmp,tmp,tmp,MSE_x(t,2)]=solve_perm_scale(kat_rao(C2,A2)*b2,x);  
    
       
%    %------------  PARAFAC_RLST ----------------------------------
%    % update the slice xu observed N steps ago 
%    % (only useful if wind=='trunc', if wind=='exp' xu is not used by PARAFAC_RLST anyway)
%    if strcmp(wind,'trunc')==1
%         xu = Xm(:,end-N+1);
%         Xm = [Xm(:,2:end),x];
%         bu = B1(end-N+1,:).';
%         tic;
%         [b1,A1,C1,P1,Q1,R1,Z1]=parafac_rlst(x,xu,bu,A1,P1,Q1,R1,Z1,wind,L,N,Niter);
%         time(t,1)=toc;
%         B1 = [B1(2:end,:); b1.'];   % matrix growing with time 
%    elseif strcmp(wind,'exp')==1
%         xu=[];   % void or anything else since it is not used
%         bu=[];   % void or anything else since it is not used
%         tic;
%         [b1,A1,C1,P1,Q1,R1,Z1]=parafac_rlst(x,xu,bu,A1,P1,Q1,R1,Z1,wind,L,N,Niter);
%         time(t,1)=toc;
%    end 
%   [tmp,tmp,tmp,MSE_A(t,1)]=solve_perm_scale(A1,A);
%   [tmp,tmp,tmp,MSE_C(t,1)]=solve_perm_scale(C1,C);
%   [tmp,tmp,tmp,MSE_x(t,1)]=solve_perm_scale(kat_rao(C1,A1)*b1,x);  
%   
  

  %------------Benchmark: Batch ALS ---------------------------------------
  Xw=[L*Xw,x];   % ALS on augmented tensor after weighting old slices
  X=permute(reshape(Xw,I,K,size(Xw,2)),[1 3 2]); % new tensor with appended slice
  tic
  B_init=[B_ALS;(pinv(kat_rao(C_ALS,A_ALS))*x).'];
  [A_ALS,B_ALS,C_ALS]=cp3_alsls(X,R,[],[],[],[],[],[],[],A_ALS,B_init,C_ALS);
  time(t,3)=toc; 
  [tmp,tmp,tmp,MSE_A(t,3)]=solve_perm_scale(A_ALS,A);
  [tmp,tmp,tmp,MSE_C(t,3)]=solve_perm_scale(C_ALS,C);
  [tmp,tmp,tmp,MSE_x(t,3)]=solve_perm_scale(kat_rao(C_ALS,A_ALS)*B_ALS(end,:).',x);
   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
    
end     %%%%%%% END of TRACKING SIMULATION  %%%%%%%%%%%%%%%%%



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% PLOT PERFORMANCE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Plot evolution of MSE of A vs tracking steps
   figure
   semilogy(MSE_A(:,1),'b');hold on
   semilogy(MSE_A(:,2),'r');
   semilogy(MSE_A(:,3),'g');
   legend('PARAFAC RLST','PARAFAC SDT','Batch ALS');
   grid on;
   xlabel('time');
   ylabel('MSE of A');
   title('Evolution of MSE of A')

% Plot evolution of MSE of x vs tracking steps
   figure
   semilogy(MSE_x(:,1),'b');hold on
   semilogy(MSE_x(:,2),'r');
   semilogy(MSE_x(:,3),'g');
   grid on;
   xlabel('time');
   ylabel('MSE of x');
   legend('PARAFAC RLST','PARAFAC-SDT','Batch ALS');
   title('Evolution of MSE of x')
   
 % Plot evolution of MSE of C vs tracking steps
   figure
   semilogy(MSE_C(:,1),'b');hold on
   semilogy(MSE_C(:,2),'r');
   semilogy(MSE_C(:,3),'g');
   grid on;
   xlabel('time');
   ylabel('MSE of C');
   legend('PARAFAC RLST','PARAFAC SDT','Batch ALS');
   title('Evolution of MSE of C')
  
% Plot evolution of CPU time vs tracking steps
   figure
   semilogy(time(:,1),'b');hold on
   semilogy(time(:,2),'r');
   semilogy(time(:,3),'g');
   grid on;
   xlabel('time');
   ylabel('CPU time');
   legend('PARAFAC RLST','PARAFAC SDT','Batch ALS');
   title('Evolution of CPU time')