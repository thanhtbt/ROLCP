function [Sen,P,D,err]=solve_perm_scale(Se,S)
%SOLVE_PERM_SCALE find permutation and scaling of columns of Se to match columns of S
%   [Se,P]=solve_perm_scale(Se,S)
%
% Given  Se = S*D*P + Noise, where:
% - Se and S are given, Se is an estimate of S,
% - D is an unknown diagonal matrix that scales the columns of S,
% - P is a permutation matrix,
% - Noise is the residual (estimation error),
% Outputs:
% - P and D 
% - the new estimate Sen=Se*P'*inv(D), such that Sen matches S
% - the relative error err=norm(Sen-S,'fro')/norm(S,'fro');
%
% Note: Se and S to be tall.
% - If Se is tall (and assumed full column rank), then P is estimated by seeking 
% for the position of the extreme values of pinv(Se)*S.
% - If Se is fat (and full row rank), then Se and S are substituted by 
% Ze=kr(Se,Se,...,Se) and Z=kr(S,S,..,S), where kr is the Khatri-Rao product
% (column-wise Kronecker product) and the number of times S and Se
% are involved in the Khatri-Rao product is the minimal number such that the 
% final matrices Z and Ze are tall (or square). Then P is estimated as in the 
% previous case, from pinv(Ze,Z)
%
% INPUTS: 
% - S   : the true matrix S
% - Se  : an estimate of S
% OUTPUTS:
% - Sen = Sen=Se*P'*inv(D)
% - P : estimate of the permutation matrix
% - D : diagonal scaling matrix
% - err= norm(Sen-S,'fro')
%
% Example:
%   I=8;R=15;
%   S=randn(I,R)+j*randn(I,R);
%   D=diag(randn(1,R)+j*randn(1,R));
%   P=eye(R);
%   P=P(:,randperm(R));
%   Se=S*D*P;
%   [Se2,P2,D2,err]=solve_perm_scale(Se,S);
%   norm(P-P2)
%   norm(D-D2)
%   err

% @Copyright July 2010
% Dimitri Nion (feedback: dimitri.nion@gmail.com)
%
% For non-commercial use only

[I,R]=size(S);
[Ie,Re]=size(Se);
if I~=Ie || R~=Re
    error('permsolve:InvalidSize',['Input matrices must have the same size']);
end
Sin=S;
% Deal with the case where matrices are fat. 
% Build S=kr(S,S,...,S) and Se=kr(Se,Se,...,Se) to get tall matrices
fat_flag=0;
if I<R
    fat_flag=1;
    Sein=Se;
    while size(Se,1)<R
        Se=kr(Se,Se);
        S=kr(S,S);
    end
end
% Normalization of all columns
for r=1:R
Se(:,r) = Se(:,r) / norm(Se(:,r));
S(:,r) = S(:,r) / norm(S(:,r));
end
prod_init=abs(pinv(Se)*S);
prod=prod_init;
c1=1:R;
c2=1:R;
p1=c1;
p2=c2;
for r=1:R
    [row,col]=find(prod==max(max(prod)));
    row=row(1);col=col(1);
    p1(r)=c1(col);
    p2(r)=c2(row);
    c1(col)=[];
    c2(row)=[];
    prod(row,:)=[];  
    prod(:,col)=[];
end
[temp p2]=sort(p2,'ascend');
perm_col=p1(p2);
% permutation matrix P
P=eye(R);
P=P(:,perm_col);
% permuted estimate
if fat_flag==0
Se(:,perm_col)= Se;
else
Sein(:,perm_col)= Sein;
Se=Sein;
end

% Now that the permutation has been found, deal with the scaling ambiguity
scal=zeros(1,R); % vectors to store scaling factors
for r=1:R
    scal(r)=(Sin(:,r)'*Sin(:,r))/(Sin(:,r)'*Se(:,r));
end
Sen=Se*diag(scal);
D=diag(1./scal);
err= norm(Sen-Sin,'fro');

%******************************************************************************
function C = kr(A,B)
%KR Khatri-Rao product.
%   kr(A,B) is the Khatri-Rao product of two matrices A and B, of 
%   dimensions I-by-R and J-by-R respectively. The result is an I*J-by-R
%   matrix formed by the columnwise Kronecker products, i.e.
%
%      kr(A,B) == [kron(A(:,1),B(:,1)) ... kron(A(:,K),B(:,K))].
%
%   See also kron.

%   Copyright 2010
%   Version: 07-07-10
%   Authors: Dimitri Nion (dimitri.nion@gmail.com),

I=size(A,1); R1=size(A,2);
J=size(B,1); R2=size(B,2);
if R1~=R2
    error('kr:ColumnMismatch',['Input matrices must have the same ' ...
          'number of columns.']);
end       
C=zeros(I*J,R1);
for j=1:R1
    C(:,j)=reshape(B(:,j)*A(:,j).',I*J,1);
end