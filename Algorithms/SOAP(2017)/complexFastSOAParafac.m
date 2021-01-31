function [b,A1,C1,H1,Hinv,Rinv] = complexFastSOAParafac(x,A0,C0,H0,Hinv,Rinv,I,K,R,t)
% Fast adaptive PARAFAC using second-order stochastic gradient decent method
%
% Synopsis: [b,A1,C1,H,Hinv,Rinv] = complexFastSOAParafac(x,A0,C0,H,Hinv,Rinv,I,K,R)
%
% Input: x      = new data in colum format
%        A0     = loading matrix A in previous snapshot
%        C0     = loading matrix C in previous snapshot
%        H      = Khatri-Rao product of C0 and A0
%        Hinv   = pseudo-inverse of H
%        Rinv   = inverse of matrix R, see the paper for more detail
%        I,K,R  = dimensions of matrix A and C, IxR and KxR respectively
%
% Ouput: b = estimated row of matrix B
%        A1     = estimated loading matrix A 
%        C1     = estimated loading matrix C 
%        H      = updated version of H
%        Hinv   = updated version of Hinv
%        Rinv   = updated version of R

% Implemented by Nguyen Viet-Dung, Karim Abed-Meraim, Nguyen Linh-Trung
% Last modified 25 April 2016
% dbstop if error
%% Step 1: estimate "coarse" b^T
lambda = 0.8;

bT = Hinv * x;

%% Step 2: fix b, update H
u = (1/lambda)*(Rinv*bT); % temp variable
v = (1/lambda)*(bT'*Rinv); % temp variable
bet = 1 + (1/lambda)*bT'*Rinv*bT;
Rinv = (1/lambda)*Rinv - (1/bet)*(u*v);

z = Rinv'*bT;
d = (x- H0*bT);
H1 = H0 + d*z';
H1_ = H1;

%% Step 3: update one column of H
A1 = zeros(I,R); 
C1 = zeros(K,R);
for r = 1:R
    Hr = reshape(H1(:,r),I,K);   % ar is the left principal sing vector and conj(cr) the left one
    c1 = Hr'*A0(:,r);    
    C1(:,r) = conj(c1);     
    a1 = Hr*c1;
    A1(:,r) = a1/norm(a1,'fro');     % Normalize it because sing value was included in c1
end


%% Step 4: Re-estimate b^T
% circular process
if (mod(t,R)~=0)
        ii = mod(t,R);
else 
         ii = R;
end

Hii = reshape(H1(:,ii),I,K);   % ar is the left principal sing vector and conj(cr) the left one
cii = Hii'*A0(:,ii);
% cii = cii./norm(cii,'fro');
aii = Hii*cii;
aii = aii/norm(aii,'fro');     % Normalize it because sing value was included in c1
H1(:,ii) = kron(conj(cii),aii);

% caluclate pseudoinverse
delta = kron(conj(cii),aii) - H1_(:,ii);
Hinv = rank2update(H0,Hinv,d,conj(z),delta,ii);

bT = Hinv*x;
b = bT.';
end

function P = pinv_update(A,P,c,d)
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
% if nargin < 4;error('Not enough input arguments.');
% elseif nargin==4;idcase='0';end
% K=size(A,1);
% R=size(A,2);
% 
% %---------------- idcase=0 -------------------------
% if strcmp(idcase,'0') 
% % Find the case between 1 and 7 by tests on the quantities
%       bet = 1+d'*P*c;
%       u    =c-A*(P*c);
%       v    =d'-(d'*P)*A;
%       if (norm(u)>1e-6) && (norm(v)>1e-6)  
%         idcase='1';
%         disp('case 1 has been detected')
%       elseif (norm(u)<1e-6) && (norm(v)>1e-6) && (abs(bet)<1e-6)
%         idcase='2';
%         disp('case 2 has been detected')
%       elseif (norm(u)<1e-6) && (norm(v)>1e-6) && (abs(bet)>1e-6)
%         idcase='3';
%         disp('case 3 has been detected')
%       elseif (norm(u)>1e-6) && (norm(v)<1e-6) && (abs(bet)<1e-6)
%         idcase='4';
%         disp('case 4 has been detected')
%       elseif (norm(u)>1e-6) && (norm(v)<1e-6) && (abs(bet)>1e-6)
%         idcase='5';
%         disp('case 5 has been detected')
%       elseif (norm(u)<1e-6) && (norm(v)<1e-6) && (abs(bet)<1e-6)
%         idcase='6';
%         disp('case 6 has been detected')
%       elseif (norm(u)<1e-6) && (norm(v)<1e-6) && (abs(bet)>1e-6)
%         idcase='7';
%         disp('case 7 has been detected')
%       end
% end
%       
% 
% %------------------------------------------------------------------------

      k = P*c;        % column vector (Rx1)
      h = d'*P;       % row vector (1xK)
      bet = 1+d'*k;    % scalar
      u = c-A*k;
      nu = u'*u;
      nh = h*h';
      
      s2 = nh*nu+bet*bet';
      z2 = P*h';
      p2 = - (nu/conj(bet))*z2 - k;
      q2h = - (nh/conj(bet))*u' - h;      
      P = P + ((1/conj(bet))*z2)*u' - ((conj(bet)/s2)*p2)*q2h;
end

function P = rank2update(A,P,c,d,m,ii)

% call rank-1 first
P = pinv_update(A,P,c,d);

% call rank-1 second, specialized
k = P*m;        % column vector (Rx1)
h = P(ii,:);       % row vector (1xK)
bet = 1+k(ii);    % scalar
u = m-A*k;
nu = u'*u;
nh = h*h';

s2 = nh*nu+bet*bet';
z2 = P*h';
p2 = - (nu/conj(bet))*z2 - k;
q2h = - (nh/conj(bet))*u' - h;

P = P + ((1/conj(bet))*z2)*u' - ((conj(bet)/s2)*p2)*q2h;    
end
