%% Shuo Zhou, Xuan Vinh Nguyen, James Bailey, Yunzhe Jia, Ian Davidson,
% "Accelerating Online CP Decompositions for Higher Order Tensors",
% (C) 2016 Shuo Zhou   
% Email: zhous@student.unimelb.edu.au

% To run the code, Tensor Toolbox is required.
% Brett W. Bader, Tamara G. Kolda and others. MATLAB Tensor Toolbox 
% Version 2.6, Available online, February 2015. 
% URL: http://www.sandia.gov/~tgkolda/TensorToolbox/

%% Update stage of OnlineCP
% input:  newX, the new incoming data tensor
%         As, a cell array contains the previous loading matrices of initX
%         Ps, Qs, cell arrays contain the previous complementary matrices
% ouputs: As, a cell array contains the updated loading matrices of initX.
%             To save time, As(N) is not modified, instead, projection of
%             newX on time mode (alpha) is given in the output
%         Ps, Qs, cell arrays contain the updated complementary matrices
%         alpha, coefficient on time mode of newX

function [ As, Ps, Qs, alpha ] = onlineCP_update( newX, As, Ps, Qs )
N = length(As)+1;
R = size(As{1},2);
dims = size(newX);
if length(dims)==N-1
    dims(end+1) = 1;
end
batchSize = dims(end);

Ks = getKhatriRaoList((As(1:N-1)));
H = getHadamard(As(1:N-1));

% update mode-N
KN = khatrirao(Ks{1}, As{1});
newXN = reshape(permute(newX, [N, 1:N-1]), batchSize, []);
alpha = newXN*KN/H;

% update mode 1 to N-1
for n=1:N-1
    newXn = reshape(permute(newX, [n, 1:n-1, n+1:N]), dims(n), []);
    Ps{n} = Ps{n}+newXn*khatrirao(alpha, Ks{n});
    Hn = H./(As{n}'*As{n});
    Qs{n} = Qs{n}+(alpha'*alpha).*Hn;
    As{n} = Ps{n}/Qs{n};

%     newXn = reshape(permute(newX, [n, 1:n-1, n+1:N]), dims(n), []);    
%     delta = khatrirao(alpha, Ks{n});
%     Ps{n} = Ps{n}+newXn*delta;
%     Qs{n} = Qs{n}+delta'*delta;
%     As{n} = Ps{n}/Qs{n};

end
end

