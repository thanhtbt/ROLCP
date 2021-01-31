%% Shuo Zhou, Xuan Vinh Nguyen, James Bailey, Yunzhe Jia, Ian Davidson,
% "Accelerating Online CP Decompositions for Higher Order Tensors",
% (C) 2016 Shuo Zhou   
% Email: zhous@student.unimelb.edu.au

% To run the code, Tensor Toolbox is required.
% Brett W. Bader, Tamara G. Kolda and others. MATLAB Tensor Toolbox 
% Version 2.6, Available online, February 2015. 
% URL: http://www.sandia.gov/~tgkolda/TensorToolbox/

%% Initialization stage of OnlineCP
% input:  initX, data tensor used for initialization
%         As, a cell array contains the loading matrices of initX
%         R, tensor rank
% ouputs: Ps, Qs, cell arrays contain the complementary matrices

function [ Ps, Qs ] = onlineCP_initial( initX, As, R )

% if As is not given, calculate the CP decomposition of the initial data
if ~exist('As')
    estInitX = cp_als(tensor(initX), R, 'tol', 1e-8);
    As = estInitX.U;
    % absorb lambda into the last dimension
    As{end} = As{end}*diag(estInitX.lambda);
end

dims = size(initX);
N = length(dims);

% for the first N-1 modes, calculte their assistant matrices P and Q
H = getHadamard(As);
Ks = getKhatriRaoList((As(1:N-1)));
for n=1:N-1
    Xn = reshape(permute(initX, [n, 1:n-1, n+1:N]), dims(n), []);
    Ps{n} = Xn*khatrirao(As{N}, Ks{n});
    Qs{n} = H./(As{n}'*As{n});
end
end

