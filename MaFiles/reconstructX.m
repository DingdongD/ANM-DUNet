%% ���ļ����ڲ���ANM-ADMM���չ������Ļָ��źź�ԭʼ�����źŷ������ɽ����Է�����
%% By Xuliang, 20230311
clc;clear;close all;

%% ������ʼ��
M = 8; % ��Ԫ��Ŀ
a = [0:M-1]'; % �������

f0 = 77e9; % Ƶ��
c = 3e8; % ����
lambda = c / f0; % ����
d = lambda / 2; % ��Ԫ���

snap = 1; % ������
fs = 1000; % ����Ƶ��
t = 1 / fs * (0 : snap - 1); % ʱ��

%% ���Ե�Ƶ�ź�����
P = 2; % �ź�Դ��Ŀ
thetas = [-28 -30 0 15 30 38 50]; % �ź�Դ����
s = (randn(P, snap) + 1j * randn(P, snap)) / sqrt(2);  
a_theta = exp(-1j * 2 * pi * d / lambda * a * sind(thetas)); % �źŵ���ʸ��
X0 = a_theta(:, 1:P) * s(1:P, :); % ��������ź�

noise_flag = 1;
if noise_flag
    snr = 10; % �����
    X = awgn(X0, snr, 'measured'); % ���������ź�
else
    X = X0;
end

para_path = 'H:\ADMM-ANM\Pyfiles\snap1_logs2\para.mat';
% para ˳��һ��Ϊ rho tau eta
para_list = load(para_path);
para_data = para_list.Para;
K = length(para_data)/3;

rho_list = para_data(1:K);
tau_list = para_data(K+1:2*K);
eta_list = para_data(2*K+1:3*K);

[X_new, T] = ADMM_ANM(X, tau_list, rho_list, eta_list, K);

%% ��ʵ�źŹ�����
theta_step = 1; % �������񲽳�
theta_grids = -90 : theta_step : 90; % ������������
sigmap2 = 0.5^2; % ��˹�˴������������ȣ�ֵԽ����Խ���Ӱ�������ڽ���ֵ�ķֱ�
True_sig = 0; % ��ʵDOA ���ø�˹���ۼ� ������sigmap2
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

