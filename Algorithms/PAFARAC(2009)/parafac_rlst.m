function [b1,A1,C1,P1,Q1,R1,Z1]=parafac_rlst(x,xu,bu,A0,P0,Q0,R0,Z0,wind,L,N,Niter)
%PARAFAC_RLST Track the parafac decomposition of a 3rd-order tensor via a Recursive Least Squares Tracking (RLST) algorithm
% [b1,A1,C1,P1,Q1,R1,Z1]=PARAFAC_RLST(x,xu,b,bu,A0,P0,Q0,R0,Z0,wind,L,N,Niter)
%
% This function is the matlab implementation of the parafac-rlst algorithm proposed in
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
%--------------------------------------------------------------------------------
% INPUTS:
% - x         : KIx1 vector (K index varying more slowly than I); 
%               x is the vectorized IxK new slice appended to the tensor.
% - xu        : KIx1 vector (K index varying more slowly than I);
%               xu is the vectorized IxK slice observed N steps ago (used only
%               if a truncated window is chosen, i.e. if wind=='trunc').
%               Suppose that X_new = [X_old, x] is the KI by J(t+1) observed 
%               tensor (in matrix format) at time t+1, obtained after appending 
%               the new slice x to all previously observed slices. 
%               Then xu has to be defined by xu = X_new(:,end-N). This is only 
%               useful if a truncated window is chosen, because we need to delete 
%               old data from the observation window.
%               Note that it is not necessary in practice to explicitely build 
%               the matrix X_new which has a growing dimension. We only need to 
%               keep track of the oldest KIx1 vector xu within the observation window.
% - bu        : Fx1 column vector; bu is the oldest row of the J(t) by F matrix B(t) 
%               within the observation window. So we have bu=B_old(end-N+1,:).' 
%               or equivalently bu=B_new(end-N,:).'
%               bu is used only if a truncated window is chosen.
% - A0 (IxF)  : Old estimate of A, from the previous tracking step
%   P0 (FxF)  : Old estimate of P, from the previous tracking step
%   Q0 (FxF)  : Old estimate of Q, from the previous tracking step.
%               Q0 is a recursive estimate of the inverse of P0
%   R0 (KIxF) : Old estimate of R, from the previous tracking step 
%   Z0 (FxKI) : Old estimate of Z, from the previous tracking step.
%               Z0 is a recursive estimate of the pseudo-inverse of R0.
% - wind      : wind='exp' for an exponential window
%               wind='trunc' for a truncated window
% - L         : lambda (forgetting factor)
% - N         : length of the window (useful for truncated window only)
% - Niter     : number of iterations of the inner loop 
%--------------------------------------------------------------------------------
% OUTPUTS:  
% - b1        : Fx1 vector; an estimate of the new row of B, i.e., B_new=[B_old;b1.']
% - A1 (IxF)
% - C1 (KxF)
% - P1 (FxF)
% - Q1 (FxF)    
% - R1 (KIxF)
% - Z1 (FxKI)
% are estimates after the current tracking step and have to be used as inputs of
% this function for the next tracking step.
%-------------------------------------------------------------------------------
% @Copyright Dimitri Nion & Nikos Sidiropoulos
% Technical University of Crete, ECE department
% Version: October 2010
% Feedback: dimitri.nion@gmail.com, nikos@telecom.tuc.gr
%
% This M-file and the code in it belongs to the holders of the copyrights.
% Do not share this code without authors permission. For non-commercial use only.
%-------------------------------------------------------------------------------

    % Dimensions of the problem
    F=size(A0,2);      % number of terms in the PARAFAC decomposition
    I=size(A0,1);      % dimension 1 (fixed)
    K=size(R0,1)/I;    % dimension 3 (fixed)
 
    
    %***************************************************************************
    % STEP 1: Get a first estimate of the new row b1 of B  
    % (Warning: b1 is a column vector)
    %***************************************************************************    
     b1 = P0*(Z0*x);
       
    %******************************************************************
    % SWITCH between exponential and truncated window
    %******************************************************************
    switch lower(wind)
        case('exp')

            %**************************************************
            % STEP 2: given b1, recursively estimate P1,Q1,R1,Z1
            %**************************************************
            % Update R1 and Z1=pinv(R1) with pseudo-inversion Lemma
            [R1,Z1]=pinv_update(L*R0,inv(L)*Z0,x,b1,'5');
            % Update P1 and Q1=inv(P1) with inversion Lemma
            [P1,Q1]=pinv_update(L*P0,inv(L)*Q0,b1,b1,'7');
            
            %**************************************************
            % STEP 3: Re-estimate b1
            %**************************************************
            b1 = P1*(Z1*x);
            % Possibly refine estimate of b1 (in general this refinement does 
            % not improve the accuracy that much so Niter=0 can also be used)
            % Inner loop with L=1 in the loop 
            % (the forgetting factor has been taken into account in STEP 2)
            for it=1:Niter  
                [R1,Z1]=pinv_update(R1,Z1,x,b1,'5');
                [P1,Q1]=pinv_update(P1,Q1,b1,b1,'7');
                b1 = P1*(Z1*x);         
            end
            
            
        case('trunc')
            %*****************************************************
            % STEP 2:  given b1, recursively estimate P1,Q1,R1,Z1
            %*****************************************************
            % Update R1 and Z1=pinv(R1) with pseudo-inversion Lemma twice
            [R_tilde,Z_tilde]=pinv_update(L*R0,inv(L)*Z0,x,b1,'5');  % --> take into account new data
            [R1,Z1]=pinv_update(R_tilde,Z_tilde,-(L^N)*xu,bu,'5');   % --> delete old data
            
            % Update P1 and Q1=inv(P1) with inversion Lemma twice
            [P_tilde,Q_tilde]=pinv_update(L*P0,inv(L)*Q0,b1,b1,'7');
            [P1,Q1]=pinv_update(P_tilde,Q_tilde,-(L^N)*bu,bu,'7');
            
            
            %**************************************************
            % STEP 3: Re-estimate b1
            %**************************************************
            b1 = P1*(Z1*x);
            % possibly refine estimate of b1 
            % (in general this refinement does not improve the accuracy that 
            % much so Niter=0 or 1 can be used)
            for it=1:Niter  % inner loop with L=1 in the loop
                [R_tilde,Z_tilde]=pinv_update(R1,Z1,x,b1,'5');
                [R1,Z1]=pinv_update(R_tilde,Z_tilde,xu,bu,'5');
                [P_tilde,Q_tilde]=pinv_update(P1,Q1,b1,b1,'7');
                [P1,Q1]=pinv_update(P_tilde,Q_tilde,bu,bu,'7');    
                b1 = P1*(Z1*x);
            end
                
    end
    
    
    %****************************************************************************
    % STEP 4: Estimate A1 and C1 from the fact that H1=R1*Q1=kat_rao(C1,A1)
    %
    % Track the principal left and right singular vector of the KxI matricized
    % representation of each column of H1. This is done here by a single BI-SVD step
    %****************************************************************************
    H1=R1*Q1;  % = estimate of kat_rao(C,A)
    A1=zeros(I,F); 
    C1=zeros(K,F);
    for r=1:F
        Hr1 = reshape(H1(:,r),I,K);   % ar is the left principal sing vector and conj(cr) the left one
        c1 = Hr1'*A0(:,r);
        C1(:,r) = conj(c1);  % QR not useful : this is a normalization step for vectors, and here we include the sing. value in c_est 
        a1=Hr1*c1;
        A1(:,r)= a1/norm(a1);  % Normalize it because sing value was included in c_est
    end

    
%*******************************************************************************   
function [A,P]=pinv_update(A,P,c,d,idcase)
%PINV_UPDATE Matrix inversion and pseudo-inversion Lemma for rank-1 updates
%   [A,P]=pinv_update(A,P,c,d,idcase)
% This function computes the pseudo-inverse or the inverse of a matrix A_new,
% where Anew results from a rank-1 update of A_old as follows:  
% A_new = A_old + c*d',
% where c and d are given vectors.
%
% Use conj(d) instead of d if A_new = A_old +c*d.' 
%
% This function is the implementation of the Formulas of section 3, 
% theorem 3.1.3 of the book:
% "Generalized Inverses of Linear Transformations", by S.L. Campbell and  C.D. Meyer, Jr.
% The notations of the book have been kept.
%
% The update rule explicitly links P_old=pinv(A_old) to P_new=pinv(A_new), 
% as follows:
%            pinv(A_new) = pinv(A_old+c*d') = pinv(A_old) + G
%       <=>  P_new = P_old + G
% where G depends on the following quantities:
%
%       bet = 1+d'*P_old*c
%       k    = P_old*c
%       h    = d'*P_old
%       u    =(eye(K,K)-A_old*P_old)*c
%       v    =d'*(eye(R,R)-P_old*A_old)
%
% The 7 following cases are possible according to values of the quantities 
% defined above.
%
% Important: to use the function, it is better to know which of the following 
% case fits your problem and enter the corresponding input "idcase" as a string
% between 1 and 7.
% If you don't know the case then enter idcase=0 or leave this field void and 
% the function will find your case itself, at the cost of some extra calculations.
% 
% The cases are the following:
% idcase='0' or idcase=void : you don't know your case, the function will select automatically
% idcase='1': u~=0 and v~=0
% idcase='2': u=0 and v~=0 and bet=0
% idcase='3': u=0 and v~=0 and bet~=0
% idcase='4': u~=0 and v=0 and bet=0 
% idcase='5': u~=0 and v=0 and bet~=0
% idcase='6': u=0 and v=0 and bet=0
% idcase='7': u=0 and v=0 and bet~=0  (Typical if A is square, then we have the matrix inversion Lemma)
%
% INPUTS: - A of size (K,R), the old matrix.
%         - P_old of size (R,K), the pseudo inverse of the old matrix.
%         - c (K,1) and d (R,1), the vectors of the rank-1 update.
%         - 'idcase' between '0' and '7'
% Outputs: - P is the pseudo-inverse of the new matrix
%          - A the new matrix such that Anew=Aold+c*d'
%-------------------------------------------------------------------------------
if nargin < 4;error('Not enough input arguments.');
elseif nargin==4;idcase='0';end
K=size(A,1);
R=size(A,2);

%---------------- idcase=0 -------------------------
if strcmp(idcase,'0') 
% Find the case between 1 and 7 by tests on the quantities
      bet = 1+d'*P*c;
      u    =c-A*(P*c);
      v    =d'-(d'*P)*A;
      if (norm(u)>1e-6) && (norm(v)>1e-6)  
        idcase='1';
        disp('case 1 has been detected')
      elseif (norm(u)<1e-6) && (norm(v)>1e-6) && (abs(bet)<1e-6)
        idcase='2';
        disp('case 2 has been detected')
      elseif (norm(u)<1e-6) && (norm(v)>1e-6) && (abs(bet)>1e-6)
        idcase='3';
        disp('case 3 has been detected')
      elseif (norm(u)>1e-6) && (norm(v)<1e-6) && (abs(bet)<1e-6)
        idcase='4';
        disp('case 4 has been detected')
      elseif (norm(u)>1e-6) && (norm(v)<1e-6) && (abs(bet)>1e-6)
        idcase='5';
        disp('case 5 has been detected')
      elseif (norm(u)<1e-6) && (norm(v)<1e-6) && (abs(bet)<1e-6)
        idcase='6';
        disp('case 6 has been detected')
      elseif (norm(u)<1e-6) && (norm(v)<1e-6) && (abs(bet)>1e-6)
        idcase='7';
        disp('case 7 has been detected')
      end
end
      

%------------------------------------------------------------------------

switch lower(idcase)
    case('1')    
      k    = P*c;        % column vector (Rx1)
      h    = d'*P;       % row vector (1xK)
      bet  = 1+d'*k;      % scalar
      u    = c-A*(P*c);    % column vector (Kx1)
      v    = d'-(d'*P)*A;   % row vector (1xR)
      
      A = A + c*d';
      pu = u'/(u'*u);  %=pinv(u)
      pv = v'/(v*v');  %=pinv(v)
      P = P - k*pu - pv*h + bet*pv*pu;
      
    case('2')   % u=0 and bet=0 so don't compute them
      k    = P*c;        % column vector (Rx1)
      h    = d'*P;       % row vector (1xK)
      v    = d'-(d'*P)*A;   % row vector (1xR)
      
      A = A + c*d';
      pk = k'/(k'*k);  %=pinv(k)
      pv = v'/(v*v');  %=pinv(v)
      P = P - k*(pk*P) - pv*h;
        
    case('3')   % don't compute u
      bet = 1+d'*P*c;    % scalar
      k    = P*c;        % column vector (Rx1)
      h    = d'*P;       % row vector (1xK)
      v    = d'-(d'*P)*A;   % row vector (1xR)
            
      A = A + c*d';
      
      nk=k'*k; %=norm(k)^2
      nv=v*v'; %=norm(v)^2
      p1= -( (nk/conj(bet))*v' + k);
      q1h = - ((nv/conj(bet))*k'*P+h);
      s1 = nk*nv + abs(bet)^2;
      P = P + (1/conj(bet))*v'*(k'*P) - (conj(bet)/s1)*p1*q1h;

    case('4')  % don't compute v and bet
      k    = P*c;        % column vector (Rx1)
      h    = d'*P;       % row vector (1xK)
      u    = c-A*(P*c);    % column vector (Kx1)
    
      A = A + c*d';
      pu = u'/(u'*u);  %=pinv(u)
      ph = h'/(h*h');  %=pinv(h)
      P = P - (P*ph)*h - k*pu;
        
    case('5')  % don't compute v
      
      k    = P*c;        % column vector (Rx1)
      h    = d'*P;       % row vector (1xK)
      bet  = 1+d'*k;    % scalar
      u=c-A*k;
      nu=u'*u;
      nh=h*h';
      
      s2=nh*nu+bet*bet';
      z2=P*h';
      p2= - (nu/conj(bet))*z2 - k;
      q2h = - (nh/conj(bet))*u' - h;
      
      A = A + c*d';
      P = P + ((1/conj(bet))*z2)*u' - ((conj(bet)/s2)*p2)*q2h;
    
    case('6')  % don't compute u, v and bet
      k    = P*c;        % column vector (Rx1)
      h    = d'*P;       % row vector (1xK)
      
      A = A + c*d';
      ph = h'/(h*h');  %=pinv(h)
      pk = k'/(k'*k);  %=pinv(k)
      P = P - k*(pk*P) - (P*ph)*h + (pk*(P*ph))*k*h;
      
    case('7')  % don't compute u and v
      k    = P*c;        % column vector (Rx1)
      bet = 1+d'*k;      % scalar
      h    = d'*P;       % row vector (1xK)
        
      A = A + c*d';
      P = P - ((1/bet)*k)*h;
      
    otherwise
        disp('Unknown case')
    
end  % of switch

   