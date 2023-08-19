%% ���ļ����ڲ���ANM-ADMM���չ�������DOA����
%% By Xuliang, 20230311
clc;clear;close all;

%% ������ʼ��
M = 8; % ��Ԫ��Ŀ
a = [0:M-1]'; % �������

f0 = 77e9; % Ƶ��
c = 3e8; % ����
lambda = c / f0; % ����
d = lambda / 2; % ��Ԫ���

snap = 5; % ������
fs = 1000; % ����Ƶ��
t = 1 / fs * (0 : snap - 1); % ʱ��

%% ���Ե�Ƶ�ź�����
P = 3; % �ź�Դ��Ŀ
% thetas = [-48 -38 -10 10 20 30 40 55 67 -55]; % �ź�Դ����
thetas = [-28 -35 0 15 30 38 50]; % �ź�Դ����
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

para_path = 'H:\ADMM-ANM\Pyfiles\snap5_logs2\para.mat';
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

%% MUSIC�����Խ����T���й���
[V, D] = eig(T); % ����ֵ�ֽ�
eig_value = real(diag(D)); % ��ȡ����ֵ
[B, I] = sort(eig_value, 'descend'); % ��������ֵ
EN = V(:, I(P+1:end)); % ��ȡ�����ӿռ�
    
PoutMusic = zeros(1, length(theta_grids));
for id = 1 : length(theta_grids)
    atheta_vec = exp(-1j * 2 * pi * [0:M-1]' * 1 / 2 * sind(theta_grids(id))); % ����ʸ��
    PoutMusic(id) = (abs(1 / (atheta_vec' * EN * EN' * atheta_vec))) ; % �����׼���
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
