%% DEMO: A FAST RANDOMIZED ADAPTIVE CP DECOMPOSITION FOR STREAMING TENSORS
% Author      : Le Trung Thanh
% Email       : letrungthanhtbt@gmail.com 
% Address     : University of Orleans

% Reference   : [1] L.T. Thanh, K. Abed-Meraim, N. L. Trung and A. Hafiane.
%                   "A fast randomized adaptive CP algorithm for streaming tensors". 
%                   IEEE-ICASSP, 2021.
%%
clear; clc; 

run_path;

%% Inputs
n_exp     = 10;
std_brt   = 1e-3;

tensor_dim = [100 150 1000];
R          = 10;
I          = tensor_dim(1);
J          = tensor_dim(2);
K          = tensor_dim(3);
mag        = 1e-3; % Time-varying factor
epsilon    = mag * ones(1,tensor_dim(3));
epsilon(600) = 1e-1; % Create an abrupt change at t = 600


%%  Evaluation 
PER_Our.A  = zeros(1,tensor_dim(3));
PER_Our.X  = zeros(1,tensor_dim(3));
PER_Our.B  = zeros(1,tensor_dim(3));
PER_Our.time = 0;

PER_OLSTEC.A = zeros(1,tensor_dim(3));
PER_OLSTEC.B = zeros(1,tensor_dim(3));
PER_OLSTEC.X = zeros(1,tensor_dim(3));
PER_OLSTEC.time = 0;

PER_TeCPSGD.A = zeros(1,tensor_dim(3));
PER_TeCPSGD.B = zeros(1,tensor_dim(3));
PER_TeCPSGD.X = zeros(1,tensor_dim(3));
PER_TeCPSGD.time = 0;

PER_SOAP.A = zeros(1,tensor_dim(3));
PER_SOAP.B = zeros(1,tensor_dim(3));
PER_SOAP.X = zeros(1,tensor_dim(3));
PER_SOAP.time = 0;


PER_OLCP.A = zeros(1,tensor_dim(3));
PER_OLCP.B = zeros(1,tensor_dim(3));
PER_OLCP.X = zeros(1,tensor_dim(3));
PER_OLCP.time = 0;


PER_RLST.A  = zeros(1,tensor_dim(3));
PER_RLST.X  = zeros(1,tensor_dim(3));
PER_RLST.B  = zeros(1,tensor_dim(3));
PER_RLST.time = 0;

PER_SDT.A  = zeros(1,tensor_dim(3));
PER_SDT.X  = zeros(1,tensor_dim(3));
PER_SDT.B  = zeros(1,tensor_dim(3));
PER_SDT.time = 0;


%% Main Program

for jj = 1:n_exp
    fprintf('RUN %d/%d \n',jj,n_exp)

    %% Generate True Tensor 
    Size      = tensor_dim(1:2);
    num_slide = tensor_dim(3);
    X_true    = online_tensor_generator(Size,R,num_slide,epsilon);
    
    X_data = zeros(tensor_dim);
    for ii = 1:num_slide
        X_data(:,:,ii) = X_true{1,ii};
    end
    
    % Add Noise
    Noise   = randn(tensor_dim);
    X_noise = X_data + std_brt * Noise;
 
    %% Adaptive CP Algorithms
    OPTS.TrueFactors = X_true(2,:);
    OPTS.TrueSlides  = X_true(1,:);
    
    
    %% PARAFAC_RLST (2009)
    t_start      = tic;
    [Factor,PER] = PARAFAC_RLST_Mod(X_noise,R,OPTS);
    t_end        = toc(t_start);
    PER_RLST.time = PER_RLST.time + t_end;
    fprintf('+ PARAFAC-RLST (2009): %f(s) \n',t_end)
    PER_RLST.A = PER.A + PER_RLST.A;
    PER_RLST.B = PER.B + PER_RLST.B;
    PER_RLST.X = PER.X + PER_RLST.X;
    
        
    %% PARAFAC_SDT (2009)
    t_start      = tic;
    [Factor,PER] = PARAFAC_SDT_Mod(X_noise,R,OPTS);
    t_end        = toc(t_start);
    PER_SDT.time = PER_SDT.time + t_end;
    fprintf('+ PARAFAC-SDT (2009): %f(s) \n',t_end)
    PER_SDT.A = PER.A + PER_SDT.A;
    PER_SDT.B = PER.B + PER_SDT.B;
    PER_SDT.X = PER.X + PER_SDT.X;
    
    
    %%  SOAP (2017) 
    t_start      = tic;
    [Factor,PER] = SOAP(X_noise,R,OPTS);
    t_end        = toc(t_start);
    PER_SOAP.time = PER_SOAP.time + t_end;
    fprintf('+ SOAP (2017): %f(s) \n',t_end)
     
    PER_SOAP.A = PER.A + PER_SOAP.A;
    PER_SOAP.B = PER.B + PER_SOAP.B;
    PER_SOAP.X = PER.X + PER_SOAP.X;
    
    
    %%  OLCP (2016) 
    t_start      = tic;
    [Factor,PER] = OLCP(X_noise,R,OPTS);
    t_end        = toc(t_start);
    PER_OLCP.time = PER_OLCP.time + t_end;
    fprintf('+ OLCP (2016): %f(s) \n',t_end)
     
    PER_OLCP.A = PER.A + PER_OLCP.A;
    PER_OLCP.B = PER.B + PER_OLCP.B;
    PER_OLCP.X = PER.X + PER_OLCP.X;
        
    %% OLSTEC (2019)
    clear options;
    Omega = ones(I,J,K);
    % Parameters for Algorithms
    maxepochs               = 1;
    verbose                 = 2;
    tolcost                 = 1e-8;
    permute_on              = false;
    options.maxepochs       = maxepochs;
    options.tolcost         = tolcost;
    options.permute_on      = permute_on;
    options.lambda          = 0.7;   % Forgetting paramter
    options.mu              = 0.1;   % Regualization paramter
    options.tw_flag         = 0;     % 0:Exponential Window, 1:Truncated Window (TW)
    options.tw_len          = 10;    % Window length for Truncated Window (TW) algorithm
    options.store_subinfo   = true;
    options.store_matrix    = false;
    options.verbose         = verbose;
   

    OPTS_2.TrueFactors = X_true(2,:);
    OPTS_2.TrueSlides  = X_true(1,:);
    OPTS_2.Omega_in    = Omega;
    OPTS_2.tensor_dims = [I J K];
    OPTS_2.Rank = R;
    Xinit.A = randn(I, R);
    Xinit.B = randn(J, R);
    Xinit.C = randn(K, R);
    OPTS_2.Xinit = Xinit;

    t_start = tic;
    [~,~,PER] = OLSTEC_MOD(X_noise, OPTS_2, options);
    t_end = toc(t_start);
    PER_OLSTEC.time = PER_OLSTEC.time + t_end;   
    fprintf('+ OLSTEC (2019): %f(s)\n',t_end)
    
    PER_OLSTEC.A = PER.A  + PER_OLSTEC.A;
    PER_OLSTEC.B = PER.B  + PER_OLSTEC.B;
    PER_OLSTEC.X = PER.X  + PER_OLSTEC.X;
    
    %% TeCPSGD (2015)
    clear options;
    Omega = ones(I,J,K);
    % Parameters for Algorithms
    maxepochs               = 1;
    verbose                 = 2;
    tolcost                 = 1e-8;
    permute_on              = false;
    options.maxepochs       = maxepochs;
    options.tolcost         = tolcost;
    options.permute_on      = permute_on;
    options.stepsize        = 0.1;
    options.lambda          = 0.001;  % Forgetting paramter
    options.mu              = 0.5; 
    options.tw_flag         = 0;     % 0:Exponential Window, 1:Truncated Window (TW)
    options.tw_len          = 10;    % Window length for Truncated Window (TW) algorithm
    options.store_subinfo   = true;
    options.store_matrix    = false;
    options.verbose         = verbose;
   

    OPTS_2.TrueFactors = X_true(2,:);
    OPTS_2.TrueSlides  = X_true(1,:);
    OPTS_2.Omega_in = Omega;
    OPTS_2.tensor_dims = [I J K];
    OPTS_2.Rank = R;
    Xinit.A = randn(I, R);
    Xinit.B = randn(J, R);
    Xinit.C = randn(K, R);
    OPTS_2.Xinit = Xinit;

    t_start = tic;
    [~,~,PER] = TeCPSGD_MOD(X_noise, OPTS_2, options);
    t_end = toc(t_start);
    PER_TeCPSGD.time = PER_TeCPSGD.time + t_end;   
    fprintf('+ TeCPSGD (2015): %f(s)\n',t_end)
    
    PER_TeCPSGD.A = PER.A  + PER_TeCPSGD.A;
    PER_TeCPSGD.B = PER.B  + PER_TeCPSGD.B;
    PER_TeCPSGD.X = PER.X  + PER_TeCPSGD.X;
    

    
    %% ROLCP (Our Method)
    t_start      = tic;
    [Factor,PER] = ROLCP(X_noise,R,OPTS);
    t_end        = toc(t_start);
    PER_Our.time = PER_Our.time + t_end;
    fprintf('+ Our Method: %f(s) \n',t_end)
    PER_Our.A = PER.A + PER_Our.A;
    PER_Our.B = PER.B + PER_Our.B;
    PER_Our.X = PER.X + PER_Our.X;
end

PER_Our.A = PER_Our.A / n_exp;
PER_Our.B = PER_Our.B / n_exp;
PER_Our.X = PER_Our.X / n_exp;
PER_Our.time = PER_Our.time / n_exp;

PER_OLSTEC.A = PER_OLSTEC.A / n_exp;
PER_OLSTEC.B = PER_OLSTEC.B / n_exp;
PER_OLSTEC.X = PER_OLSTEC.X / n_exp;
PER_OLSTEC.time = PER_OLSTEC.time / n_exp;

PER_TeCPSGD.A = PER_TeCPSGD.A / n_exp;
PER_TeCPSGD.B = PER_TeCPSGD.B / n_exp;
PER_TeCPSGD.X = PER_TeCPSGD.X / n_exp;
PER_TeCPSGD.time = PER_TeCPSGD.time / n_exp;


PER_SOAP.A = PER_SOAP.A / n_exp;
PER_SOAP.B = PER_SOAP.B / n_exp;
PER_SOAP.X = PER_SOAP.X / n_exp;
PER_SOAP.time = PER_SOAP.time / n_exp;


PER_OLCP.A = PER_OLCP.A / n_exp;
PER_OLCP.B = PER_OLCP.B / n_exp;
PER_OLCP.X = PER_OLCP.X / n_exp;
PER_OLCP.time = PER_OLCP.time / n_exp;

PER_RLST.A = PER_RLST.A / n_exp;
PER_RLST.B = PER_RLST.B / n_exp;
PER_RLST.X = PER_RLST.X / n_exp;
PER_RLST.time = PER_RLST.time / n_exp;


PER_SDT.A = PER_SDT.A / n_exp;
PER_SDT.B = PER_SDT.B / n_exp;
PER_SDT.X = PER_SDT.X / n_exp;
PER_SDT.time = PER_SDT.time / n_exp;

fprintf('\n Average Running Time: \n')
fprintf(' \n + ROLCP (Proposed): %f (s)', PER_Our.time);
fprintf(' \n + OLCP: %f (s)', PER_OLCP.time);
fprintf(' \n + OLSTEC: %f (s)', PER_OLSTEC.time);
fprintf(' \n + TeCPSGD: %f (s)', PER_TeCPSGD.time);
fprintf(' \n + RLST: %f (s)', PER_RLST.time);
fprintf(' \n + SDT: %f (s)', PER_SDT.time);
fprintf(' \n + SOAP: %f (s)\n', PER_SOAP.time);

%% %% PLOT RESULTS
makerSize = 14;
numbMarkers = 50;
LineWidth = 2;
set(0, 'defaultTextInterpreter', 'latex');
color   = get(groot,'DefaultAxesColorOrder');
red_o   = [1,0,0];
blue_o  = [0, 0, 1];
gree_o  = [0, 0.5, 0];
black_o = [0.25, 0.25, 0.25];

blue_n  = color(1,:);
oran_n  = color(2,:);
yell_n  = color(3,:);
viol_n  = color(4,:);
gree_n  = color(5,:);
lblu_n  = color(6,:);
brow_n  = color(7,:);
lbrow_n = [0.5350    0.580    0.2840];

%%
fig = figure;

subplot(121);
hold on;
k = 5;

d2 = semilogy(1:k:K,PER_RLST.X(1:k:end),...
    'linestyle','-','color',oran_n,'LineWidth',LineWidth);
d21 = plot(1:100:K,PER_RLST.X(1:100:end),...
 'marker','o','markersize',makerSize,...
   'linestyle','none','color',oran_n,'LineWidth',LineWidth);
d22 = semilogy(1:1,PER_RLST.X(1:1),...
    'marker','o','markersize',makerSize,...
    'linestyle','-','color',oran_n,'LineWidth',LineWidth);


d3 = semilogy(1:k:K,PER_SDT.X(1:k:end),...
    'linestyle','-','color',viol_n,'LineWidth',LineWidth);
d31 = plot(1:100:K,PER_SDT.X(1:100:end),...
 'marker','s','markersize',makerSize,...
   'linestyle','none','color',viol_n,'LineWidth',LineWidth);
d32 = semilogy(1:1,PER_SDT.X(1:1),...
    'marker','s','markersize',makerSize,...
    'linestyle','-','color',viol_n,'LineWidth',LineWidth);


d4 = semilogy(1:k:K,PER_SOAP.X(1:k:end),...
    'linestyle','-','color',black_o,'LineWidth',LineWidth);
d41 = plot(1:100:K,PER_SOAP.X(1:100:end),...
 'marker','^','markersize',makerSize,...
   'linestyle','none','color',black_o,'LineWidth',LineWidth);
d42 = semilogy(1:1,PER_SOAP.X(1:1),...
    'marker','^','markersize',makerSize,...
    'linestyle','-','color',black_o,'LineWidth',LineWidth);

d5 = semilogy(1:k:K,PER_OLCP.X(1:k:end),...
    'linestyle','-','color',gree_n,'LineWidth',LineWidth);
d51 = plot(1:100:K,PER_OLCP.X(1:100:end),...
 'marker','*','markersize',makerSize,...
   'linestyle','none','color',gree_n,'LineWidth',LineWidth);
d52 = semilogy(1:1,PER_OLCP.X(1:1),...
    'marker','*','markersize',makerSize,...
    'linestyle','-','color',gree_n,'LineWidth',LineWidth);


d6 = semilogy(1:k:K,PER_TeCPSGD.X(1:k:end),...
    'linestyle','-','color',blue_o,'LineWidth',LineWidth);
d61 = plot(1:100:K,PER_TeCPSGD.X(1:100:end),...
 'marker','v','markersize',makerSize,...
   'linestyle','none','color',blue_o,'LineWidth',LineWidth);
d62 = semilogy(1:1,PER_TeCPSGD.X(1:1),...
    'marker','v','markersize',makerSize,...
    'linestyle','-','color',blue_o,'LineWidth',LineWidth);


d7 = semilogy(1:k:K,PER_OLSTEC.X(1:k:end),...
    'linestyle','-','color',gree_o,'LineWidth',LineWidth);
d71 = plot(1:100:K,PER_OLSTEC.X(1:100:end),...
 'marker','p','markersize',makerSize,...
   'linestyle','none','color',gree_o,'LineWidth',LineWidth);
d72 = semilogy(1:1,PER_OLSTEC.X(1:1),...
    'marker','p','markersize',makerSize,...
    'linestyle','-','color',gree_o,'LineWidth',LineWidth);


d1 = semilogy(1:k:K,PER_Our.X(1:k:end),...
    'linestyle','-','color',red_o,'LineWidth',LineWidth);

d11 = plot(1:100:K,PER_Our.X(1:100:end),...
 'marker','d','markersize',makerSize,...
   'linestyle','none','color',red_o,'LineWidth',LineWidth);
d12 = semilogy(1:1,PER_Our.X(1:1),...
    'marker','d','markersize',makerSize,...
    'linestyle','-','color',red_o,'LineWidth',LineWidth);



lgd = legend([ d22 d32  d42 d52 d62 d72 d12],'\texttt{PARAFAC-RLST}','\texttt{PARAFAC-SDT}','\texttt{SOAP}','\texttt{OLCP}','\texttt{TeCPSGD}','\texttt{OLSTEC}','\texttt{ROLCP(Proposed)}');
lgd.FontSize = 16;
set(lgd, 'Interpreter', 'latex', 'Color', [0.95, 0.95, 0.95]);


xlabel('Time Index - $t$','interpreter','latex','FontSize',13,'FontName','Times New Roman');
ylabel('RE $(\mathcal{X}_{tr}, \mathcal{X}_{es})$','interpreter','latex','FontSize',13,'FontName','Times New Roman');

h1=gca;
set(gca, 'YScale', 'log')
set(h1,'FontSize',16,'XGrid','on','YGrid','on','GridLineStyle','-','MinorGridLineStyle','-','FontName','Times New Roman');
set(h1,'Xtick',0:200:K,'FontSize',16,'XGrid','on','YGrid','on','GridLineStyle',':','MinorGridLineStyle','none',...
    'FontName','Times New Roman');
set(h1,'FontSize', 24);
axis([0 K 0.7*std_brt 1e1]);
grid on;
box on;


%%
subplot(122);
hold on;
k = 5;

d2 = semilogy(1:k:K,PER_RLST.A(1:k:end),...
    'linestyle','-','color',oran_n,'LineWidth',LineWidth);
d21 = plot(1:100:K,PER_RLST.A(1:100:end),...
 'marker','o','markersize',makerSize,...
   'linestyle','none','color',oran_n,'LineWidth',LineWidth);
d22 = semilogy(1:1,PER_RLST.A(1:1),...
    'marker','o','markersize',makerSize,...
    'linestyle','-','color',oran_n,'LineWidth',LineWidth);


d3 = semilogy(1:k:K,PER_SDT.A(1:k:end),...
    'linestyle','-','color',viol_n,'LineWidth',LineWidth);
d31 = plot(1:100:K,PER_SDT.A(1:100:end),...
 'marker','s','markersize',makerSize,...
   'linestyle','none','color',viol_n,'LineWidth',LineWidth);
d32 = semilogy(1:1,PER_SDT.A(1:1),...
    'marker','s','markersize',makerSize,...
    'linestyle','-','color',viol_n,'LineWidth',LineWidth);


d4 = semilogy(1:k:K,PER_SOAP.A(1:k:end),...
    'linestyle','-','color',black_o,'LineWidth',LineWidth);
d41 = plot(1:100:K,PER_SOAP.A(1:100:end),...
 'marker','^','markersize',makerSize,...
   'linestyle','none','color',black_o,'LineWidth',LineWidth);
d42 = semilogy(1:1,PER_SOAP.A(1:1),...
    'marker','^','markersize',makerSize,...
    'linestyle','-','color',black_o,'LineWidth',LineWidth);

d5 = semilogy(1:k:K,PER_OLCP.A(1:k:end),...
    'linestyle','-','color',gree_n,'LineWidth',LineWidth);
d51 = plot(1:100:K,PER_OLCP.A(1:100:end),...
 'marker','*','markersize',makerSize,...
   'linestyle','none','color',gree_n,'LineWidth',LineWidth);
d52 = semilogy(1:1,PER_OLCP.A(1:1),...
    'marker','*','markersize',makerSize,...
    'linestyle','-','color',gree_n,'LineWidth',LineWidth);


d6 = semilogy(1:k:K,PER_TeCPSGD.A(1:k:end),...
    'linestyle','-','color',blue_o,'LineWidth',LineWidth);
d61 = plot(1:100:K,PER_TeCPSGD.A(1:100:end),...
 'marker','v','markersize',makerSize,...
   'linestyle','none','color',blue_o,'LineWidth',LineWidth);
d62 = semilogy(1:1,PER_TeCPSGD.A(1:1),...
    'marker','v','markersize',makerSize,...
    'linestyle','-','color',blue_o,'LineWidth',LineWidth);


d7 = semilogy(1:k:K,PER_OLSTEC.A(1:k:end),...
    'linestyle','-','color',gree_o,'LineWidth',LineWidth);
d71 = plot(1:100:K,PER_OLSTEC.A(1:100:end),...
 'marker','p','markersize',makerSize,...
   'linestyle','none','color',gree_o,'LineWidth',LineWidth);
d72 = semilogy(1:1,PER_OLSTEC.A(1:1),...
    'marker','p','markersize',makerSize,...
    'linestyle','-','color',gree_o,'LineWidth',LineWidth);


d1 = semilogy(1:k:K,PER_Our.A(1:k:end),...
    'linestyle','-','color',red_o,'LineWidth',LineWidth);

d11 = plot(1:100:K,PER_Our.A(1:100:end),...
 'marker','d','markersize',makerSize,...
   'linestyle','none','color',red_o,'LineWidth',LineWidth);
d12 = semilogy(1:1,PER_Our.A(1:1),...
    'marker','d','markersize',makerSize,...
    'linestyle','-','color',red_o,'LineWidth',LineWidth);

xlabel('Time Index - $t$','interpreter','latex','FontSize',13,'FontName','Times New Roman');
ylabel('RE $(\mathbf{A}_{tr}, \mathbf{A}_{es})$','interpreter','latex','FontSize',13,'FontName','Times New Roman');

h2=gca;
set(gca, 'YScale', 'log')
set(h2,'FontSize',16,'XGrid','on','YGrid','on','GridLineStyle','-','MinorGridLineStyle','-','FontName','Times New Roman');
set(h2,'Xtick',0:200:K,'FontSize',16,'XGrid','on','YGrid','on','GridLineStyle',':','MinorGridLineStyle','none',...
    'FontName','Times New Roman');
set(h2,'FontSize', 24);
axis([0 K 0.7*std_brt 1e1]);
grid on;
box on;

set(fig, 'units', 'inches', 'position', [0.3 0.5 13 6]);

