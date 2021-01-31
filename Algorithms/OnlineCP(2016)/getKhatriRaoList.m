%% Shuo Zhou, Xuan Vinh Nguyen, James Bailey, Yunzhe Jia, Ian Davidson,
% "Accelerating Online CP Decompositions for Higher Order Tensors",
% (C) 2016 Shuo Zhou   
% Email: zhous@student.unimelb.edu.au

% To run the code, Tensor Toolbox is required.
% Brett W. Bader, Tamara G. Kolda and others. MATLAB Tensor Toolbox 
% Version 2.6, Available online, February 2015. 
% URL: http://www.sandia.gov/~tgkolda/TensorToolbox/

%% Get a list of Khatri-Rao products
% input:  As, a cell array contains N-1 matrices, {A(1), A(2), ..., A(N-1)}
% ouputs: Ks, a list of khatri-rao products, the i-th elment of it is
%         khatriRao(A(N), ..., A(i+1), A(i-1), ..., A(1))

function [ Ks ] = getKhatriRaoList( As )

N = length(As)+1;
lefts = {As{N-1}};
rights = {As{1}};
if N>3
    for n=2:N-2
        lefts{n} = khatrirao(lefts{n-1}, As{N-n});
        rights{n} = khatrirao(As{n}, rights{n-1});
    end
end

Ks{1} = lefts{N-2};
Ks{N-1} = rights{N-2};

if N>3
    for n=2:N-2
        Ks{n} = khatrirao(lefts{N-n-1}, rights{n-1});
    end
end
end

