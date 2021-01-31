%% Shuo Zhou, Xuan Vinh Nguyen, James Bailey, Yunzhe Jia, Ian Davidson,
% "Accelerating Online CP Decompositions for Higher Order Tensors",
% (C) 2016 Shuo Zhou   
% Email: zhous@student.unimelb.edu.au

% To run the code, Tensor Toolbox is required.
% Brett W. Bader, Tamara G. Kolda and others. MATLAB Tensor Toolbox 
% Version 2.6, Available online, February 2015. 
% URL: http://www.sandia.gov/~tgkolda/TensorToolbox/

%% calculate the Hadamard product of a list of matrices
% input:  As, a cell array contains N-1 matrices, {A(1), A(2), ..., A(N-1)}
% ouputs: Had, the Hadard product of As

function [ Had ] = getHadamard( As )

Had = [];
for n=1:length(As)
    if isempty(Had)
        Had = As{n}'*As{n};
    else
        Had = Had.*(As{n}'*As{n});
    end
end

end

