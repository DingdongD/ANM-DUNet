%% 本文件用于测试ANM-ADMM深度展开网络的DOA估计
%% By Xuliang, 20230311
clc;clear;close all;

%% 参数初始化
M = 8; % 阵元数目
a = [0:M-1]'; % 导向序号

f0 = 77e9; % 频率
c = 3e8; % 光速
lambda = c / f0; % 波长
d = lambda / 2; % 阵元间距

snap = 5; % 快拍数
fs = 1000; % 采样频率
t = 1 / fs * (0 : snap - 1); % 时间

%% 线性调频信号生成
P = 3; % 信号源数目
% thetas = [-48 -38 -10 10 20 30 40 55 67 -55]; % 信号源方向
thetas = [-28 -35 0 15 30 38 50]; % 信号源方向
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

para_path = 'H:\ADMM-ANM\Pyfiles\snap5_logs2\para.mat';
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

%% MUSIC方法对解出的T进行估计
[V, D] = eig(T); % 特征值分解
eig_value = real(diag(D)); % 提取特征值
[B, I] = sort(eig_value, 'descend'); % 排序特征值
EN = V(:, I(P+1:end)); % 提取噪声子空间
    
PoutMusic = zeros(1, length(theta_grids));
for id = 1 : length(theta_grids)
    atheta_vec = exp(-1j * 2 * pi * [0:M-1]' * 1 / 2 * sind(theta_grids(id))); % 导向矢量
    PoutMusic(id) = (abs(1 / (atheta_vec' * EN * EN' * atheta_vec))) ; % 功率谱计算
end
Pmusic_db = PoutMusic/max(PoutMusic);

[Phi, Val] = rootmusic(T, P, 'corr');
Phis = Phi / 2 / pi ;
estimated_theta = round(asind(-Phis * lambda / d));
        
figure(1);
plot(theta_grids, (Pmusic_db), 'bd:', 'LineWidth', 1.5); hold on;
plot(theta_grids, (True_sig), 'r.-','LineWidth', 1.5); hold on;
xlabel('Theta(deg)'); ylabel('Amplitude');
legend('Estimated', 'True');
grid minor;
