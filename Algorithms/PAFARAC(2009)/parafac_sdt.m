function [b1,A1,C1,W1,Wi1,V1,S1,U1]=parafac_sdt(x,A0,W0,Wi0,V0,S0,U0,L)
%PARAFAC_SDT Track the parafac decomposition of a 3rd-order tensor via a Simultaneous Diagonalization Tracking (SDT)
% [b1,A1,C1,W1,Wi1,V1,S1,U1]=parafac_sdt(x,A0,W0,Wi0,V0,S0,U0,L)
%
% This function is the matlab implementation of the parafac-sdt algorithm proposed in
% 
% D. Nion and N. D. Sidiropoulos, "Adaptive Algorithms to Track the PARAFAC Decomposition of a 
% Third-Order Tensor", IEEE. Trans. on Signal Processing, vol. 57, no. 6, June 2009 
%
% If you make use of this code, please cite this paper.
%
% PROBLEM STATEMENT:
% Consider a 3rd-order tensor X(t) of dimensions IxJ(t)xK at time t, where the 
% second dimension is growing with time. 
% We are given the loading matrices A(t), B(t) and C(t), of size IxF, J(t)xF 
% and KxF, respectively, of its parafac 
% At time t+1, a new slice is appended in the second dimension of the observed
% tensor, such that X(t+1) is of size I by J(t)+1 by K.
% The purpose is to find estimates of the loading matrices A(t+1), B(t+1) and 
% C(t+1) of the parafac decomposition of X(t+1), where:
% - A(t+1) is of size IxF and all its entries may be different from the entries of A
% - C(t+1) is of size KxF and all its entries may be different from the entries of C
% - B(t+1) is of size J(t)+1 by F and is obtained from B(t) by appending a new
% unknown row b1, such that B_new=[B_old;b1.'], i.e., B has a time-shift structure.
%
% The key feature of this algorithm is that the loadings at time t+1 can be
% recursively updated from their values at time t, such that the global complexity
% is very low; it is not needed to compute the decomposition of X(t+1) by a batch
% algorithm.
%
% IMPORTANT:
% This code performs a SINGLE tracking step and has to be called each time a new
% slice is appended to the observed tensor, i.e., each time a new KIx1 vector x 
% becomes available. It means that the output of this algorithm at tracking step t
% will be used as inputs at tracking step t+1. See demo file for details.
%-------------------------------------------------------------------------------
% INPUTS:
% - x         : KIx1 vector (K index varying more slowly than I); 
%               x is the vectorized IxK new slice appended to the tensor.
% - A0 (IxF)  : Old estimate of A, from the previous tracking step
% - W0 (FxF)  : Old estimate of W, from the previous tracking step.
% - Wi0 (FxF) : Old estimate of Wi0, which is a recursive estimate of the inverse of W
% - V0 (NxF), 
%   S0 (FxF), 
%   U0(KIxF)  : matrices useful to track the SVD of the KIxN matrix unfolding of
%               the observed tensor, where N is the number of the most recent 
%               slices taking into account in the truncated window
% - L         : lambda (forgetting factor)
%-------------------------------------------------------------------------------
% OUTPUTS:  
% - b1        : Fx1 vector which is an estimate of the new row of B, i.e., B_new=[B_old;b1.']
% - A1 (IxF)  : update of A
% - C1 (KxF)  : update of C
% - W1 (FxF)  : update of W
% - Wi1(FxF)  : update of Wi  
% - V1 (NxF),
%   S1 (FxF),
%   U1 (KIxF) : updates of U,S,V
% --> These new estimates for the current tracking step, have to be used as 
% inputs of this function for the next tracking step
%-------------------------------------------------------------------------------
% @Copyright Dimitri Nion & Nikos Sidiropoulos
% Technical University of Crete, ECE department
% Version: October 2010
% Feedback: dimitri.nion@gmail.com, nikos@telecom.tuc.gr
%
% This M-file and the code in it belongs to the holders of the copyrights.
% Do not share this code without authors permission. For non-commercial use only.
%-------------------------------------------------------------------------------

   % size of the problem
    F=size(A0,2);   % number of terms in the PARAFAC decomposition (is supposed to be fixed during the whole tracking procedure)
    I=size(A0,1);   % First dimension (fixed)
    K=length(x)/I; % Third dimension (fixed)

   %****************************************************************************
   % STEP 1 : Track the SVD of the KIxN matrix unfolding of the observed tensor 
   % where only the N most recent slices are considered
   % The SWASVD (Sliding Window Adaptive SVD) algorithm by Badeau et al is used.
   %****************************************************************************
   [U1,S1,V1,E1] = SWASVD(x,U0,S0,V0,L,'right');

   %****************************************************************************
   % STEP 2: update the matrix W and its inverse Wi in a recursive way. 
   % W is the F by F matrix such that kat_rao(C,A)=E*W, where 
   % E=U*S and [U,S,V]=svd_tracking(X)
   %****************************************************************************                    
   Z = V1(1:end-1,:)'*V0(2:end,:);
   v0 = V0(1,:);
   v1 = V1(end,:);
   W1 = (L^-0.5) * Z * ( eye(F) + (v0'*v0)/(1-v0*v0') ) * W0;    % Matrix inversion Lemma
   Wi1= (L^0.5) * Wi0 * Z' * ( eye(F) + (v1'*v1)/(1-v1*v1') );   % Matrix inversion Lemma
   b1 = Wi1*v1';   % estimate the new row of B (is a column vector here)
      
   %****************************************************************************
   % STEP 3: update A and C
   %****************************************************************************
   % First update the khatri rao product
    H1 = E1*W1;
    
    % Then perform a single step of bi-iteration for eacch column of H1
    A1=zeros(I,F); 
    C1=zeros(K,F);
    for r=1:F
        Hr1 = reshape(H1(:,r),I,K);   % ar is the left principal sing vector and conj(cr) the left one
        c1=Hr1'*A0(:,r);
        C1(:,r)= conj(c1);     
        a1=Hr1*c1;
        A1(:,r)= a1/norm(a1,'fro');     % Normalize it because sing value was included in c1
    end

    % The equivalent batch code is
%     A1=zeros(I,F); 
%     C1=zeros(K,F);
%     for r=1:F
%         Hr1=reshape(H1(:,r),I,K);  % this is an estimate of the rank-1 matrix ar*cr.'
%         [a1,s1,c1]=svds(Hr1,1);    % compute only the first singular vectors
%         A1(:,r)= a1*sqrt(s1);
%         C1(:,r)= conj(c1)*sqrt(s1);
%     end
    
%***********************************************************************************************************
function [U_new,S_new,V_new,E_new] = SWASVD(x_new,U_old,S_old,V_old,F,direction)
% SWASVD algorithm to track the SVD of a data matrix updated with a truncated window
% Reference: "Sliding Window Adaptive SVD Algorithms" by R. Badeau, G. Richard and B. David
%-----------------------------------------------------------------------------------------------------------
% m-code by Dimitri Nion and Nikos Sidiropoulos
%-----------------------------------------------------------------------------------------------------------
% How to use:
% Suppose X_old=[x(t-1), F^(1/2)*x(t-2), ..., F^(L-1)/2*x(t-L)]   (1) 
% is the old observed weighted matrix, of size NxL. Each column vector x
% has length N and L vectors are collected.
% F is the forgetting factor (not in the original paper but added here as
% an option).
% U_old (NxR) holds the estimated left R principal singular vectors of X_old.
% V_old (LxR) holds the estimated right R principal singular vectors of X_old.
% S_old (RxR) holds the estimated singular values (becomes diagonal for an iterative use of this function)
%
% Suppose X_new=[x(t),F^(1/2)*x(t-1), ..., F^(L-1)/2*x(t-L+1)]   (2), 
% which is linked to X_old by a sliding window as follows:
% [X_new, F^(L/2)*x(t-L)]=[x(t),F^(1/2)*X_old],
% where x(t) is the new observed vector, i.e. x_new as input of the function
% direction: 
% 'left' : if the most recent vectors is appended from the left as in (1)and (2)
% 'right': if time is going from left to right (new vector appended from the right)
% Outputs: estimated U, S and V for X_new.
%-------------------------------------------------------------------------------------------
switch lower(direction)
    case('left')
% First iteration: update V
h = U_old'*x_new;
B = [h';F^(1/2)*V_old*S_old'];
[V_new,Rb]=qr(B(1:end-1,:),0);

% Second iteration: update U
xo = x_new - U_old*h;
A = U_old*Rb'+xo*V_new(1,:);
[U_new,S_new]=qr(A,0);

    case('right')
    % First iteration: update V
    h = U_old'*x_new;
    B = [F^(1/2)*V_old*S_old';h'];
    [V_new,Rb]=qr(B(2:end,:),0);

    % Second iteration: update U
    xo = x_new - U_old*h;
    E_new = U_old*Rb'+xo*V_new(end,:);
    [U_new,S_new]=qr(E_new,0);
end