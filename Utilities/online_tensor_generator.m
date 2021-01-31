function X_stream = online_tensor_generator(Size,Rank,num_slides,epsilon)
%% At each time instant t
% X_stream{1,:}: Slides of Tensor
% X_stream{2,:}: Factors of Tensor
% X_t = A_t * diag(c_t) * B_t';
% factors are changing slowly or static
% A_t = A_{t-1} + epsilon*randn(Size);
% B_t = B_{t-1} + epsilon*randn(Size);


%% Main Algorithms
L = length(Size);
Factors = cell(1,L);
for n = 1:L
    Factors{n} = randn(Size(n),Rank);
end
X_stream{2,1} = Factors;
X_stream{1,1} = tensor_construction(Factors,Rank);

for t = 2 : num_slides
    for n = 1:L
        Factors{n} = Factors{n} + epsilon(t)*randn(Size(n),Rank);
    end
        
    X_stream{2,t} = Factors;
    X_stream{1,t} = tensor_construction(Factors,Rank);
end

end

function T = tensor_construction(Factors,Rank)
coeff = randn(Rank,1);
T = Factors{1} * diag(coeff) * (Factors{2})';
end