%% Shuo Zhou, Xuan Vinh Nguyen, James Bailey, Yunzhe Jia, Ian Davidson,
% "Accelerating Online CP Decompositions for Higher Order Tensors",
% (C) 2016 Shuo Zhou   
% Email: zhous@student.unimelb.edu.au

% To run the code, Tensor Toolbox is required.
% Brett W. Bader, Tamara G. Kolda and others. MATLAB Tensor Toolbox 
% Version 2.6, Available online, February 2015. 
% URL: http://www.sandia.gov/~tgkolda/TensorToolbox/

%% Generate a data tensor
% input:  I, a vector contains the dimensionality of the data tensor
%         R, rank of data tensor
%         SNR, Noise level in dB, inf for noiseless tensor
% ouputs: X, the output data tensor

function [ X ] = generateData( I, R, SNR )

if ~exist('SNR')
    SNR = 30; 
end

A = arrayfun(@(x) randn(x, R), I, 'uni', 0);
X = ktensor(A(:));

% add noise
normX = norm(X);
if ~isinf(SNR)
    noise = randn(I);
    sigma = (10^(-SNR/20))*(norm(X))/norm(tensor(noise));
    X = double(full(X))+sigma*noise;
end
end

