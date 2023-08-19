%% 本文件用于测试ANM-ADMM深度展开网络的恢复信号和原始基带信号分析【可解释性分析】
%% By Xuliang, 20230311
clc;clear;close all;

%% 参数初始化
M = 8; % 阵元数目
a = [0:M-1]'; % 导向序号

f0 = 77e9; % 频率
c = 3e8; % 光速
lambda = c / f0; % 波长
d = lambda / 2; % 阵元间距

snap = 1; % 快拍数
fs = 1000; % 采样频率
t = 1 / fs * (0 : snap - 1); % 时间

%% 线性调频信号生成
P = 2; % 信号源数目
thetas = [-28 -30 0 15 30 38 50]; % 信号源方向
s = (randn(P, snap) + 1j * randn(P, snap)) / sqrt(2);  
a_theta = exp(-1j * 2 * pi * d / lambda * a * sind(thetas)); % 信号导向矢量
X0 = a_theta(:, 1:P) * s(1:P, :); % 无噪基带信号

noise_flag = 1;
if noise_flag
    snr = 10; % 信噪比
    X = awgn(X0, snr, 'measured'); % 完整基带信号
else
    X = X0;
end

para_path = 'H:\ADMM-ANM\Pyfiles\snap1_logs2\para.mat';
% para 顺序一般为 rho tau eta
para_list = load(para_path);
para_data = para_list.Para;
K = length(para_data)/3;

rho_list = para_data(1:K);
tau_list = para_data(K+1:2*K);
eta_list = para_data(2*K+1:3*K);

[X_new, T] = ADMM_ANM(X, tau_list, rho_list, eta_list, K);

%% 真实信号估计谱
theta_step = 1; % 遍历网格步长
theta_grids = -90 : theta_step : 90; % 遍历网格区间
sigmap2 = 0.5^2; % 高斯核带宽，控制主瓣宽度，值越大宽带越大会影响两个邻近峰值的分辨
True_sig = 0; % 真实DOA 采用高斯谱累加 依赖于sigmap2
for theta_idx = 1 : P
    True_sig = True_sig + exp(-(theta_grids - thetas(theta_idx)).^2 / 2 / sigmap2);
end

arrayNum = 1:M;
snapNum = 1 : snap;
figure(1);
X_new = abs(X_new / max(max(X_new)))*1;
X = abs(X / max(max(X)))*1;
imagesc(snapNum,arrayNum, abs(X_new)); colormap('winter');colorbar;caxis([0 1]);
xlabel('Snap');ylabel('ArrayNum');
figure(2);
imagesc(snapNum,arrayNum, abs(X));colormap('winter');colorbar;caxis([0 1]);
xlabel('Snap');ylabel('ArrayNum');

