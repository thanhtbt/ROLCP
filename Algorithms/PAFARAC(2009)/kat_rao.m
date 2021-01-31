function C = kat_rao(A,B)
%--------------------------------------------------------------------------------------------------
% Calculate the Kathri Rao product of 2 matrices
%--------------------------------------------------------------------------------------------------
        I=size(A,1);R1=size(A,2);
        J=size(B,1);R2=size(B,2);
        if R1~=R2
            error('Input matrices must have the same number of columns for Khatri Rao product')
        end
        
        C=zeros(I*J,R1);
        for j=1:R1
            C(:,j)=reshape(B(:,j)*A(:,j).',I*J,1);
        end