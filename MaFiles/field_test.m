%% 本文件用于测试东大实测静态车载场景下的RA图


clc;clear;close all;
f0 = 77e9; % 频率
c = 3e8; % 光速
lambda = c / f0; % 波长
d = lambda / 2; % 阵元间距

range_data = load('range_fft.mat');
range_data = range_data.range_fft;

angle_axis = [-90:90]; % 角度索引
max_range = 12.4992;
range_axis = [0:max_range/(size(range_data,1)-1):max_range];
P = 2;
RAmap = zeros(size(range_data,1), length(angle_axis));

para_path = 'H:\ADMM-ANM\Pyfiles\snap3_logs1\para.mat';
para_list = load(para_path);
para_data = para_list.Para;
K = length(para_data)/3;

rho_list = para_data(1:K);
tau_list = para_data(K+1:2*K);
eta_list = para_data(2*K+1:3*K);

params.mode = "APES";
params.iter_num = 100;
params.threshold = 1e-8;
a = [0:size(range_data,2)-1]';
A = exp(-1j * 2 * pi * d / lambda * a * sind(angle_axis)); % 构造过完备字典
M = size(range_data,2);

flag = 0;
pic_num = 1;
for id = 1:size(range_data,1)
%     X = squeeze(range_data(id, :, 1)).';
% 
    X = squeeze(range_data(id, :, 1:128));
    X = sum(X,2);
    RX = X*X';
    
    if flag
        [V, D] = eig(RX); % 特征值分解
        figure(1); % 特征值分布的可视化
        
        SP = V(:, M-P+1:M);
        EN = V(:, 1:M-P);
        s_eigen_fft = fft(SP);
        n_eigen_fft = fft(EN);
        plot(1:M,abs(s_eigen_fft), 'ks-','LineWidth',1.5); hold on;
        plot(1:M,abs(n_eigen_fft), 'r','LineWidth',1);
        legend('Signal Space', 'Noise Space','location','NorthWest');

%         plot(diag((D)),'kd-');
        xlabel('Number of Eigenvalues'); ylabel('Eigenvalues');
        title(['目标序号：',num2str(id)]);
        
        drawnow; % 刷新屏幕
        F = getframe(gcf); % getframe捕获坐标区或图窗作为影片帧，gcf返回当前figure的句柄
        I = frame2im(F); % frame2im返回与影片帧关联的图像数据。输出一个三维矩阵
        [I,map]=rgb2ind(I,256);    %rgb2ind将rgb图像转化成索引(index)图像。
        if pic_num == 1
            imwrite(I,map,'test1.gif','gif', 'Loopcount',inf,'DelayTime',0.1);
        else
            imwrite(I,map,'test1.gif','gif','WriteMode','append','DelayTime',0.1);
        end
        pic_num = pic_num + 1;
        
        pause(0.1);
        hold off
    else
        % 下面是MUSIC的方法
        Pout = DOA_MUSIC(RX, 2, angle_axis);
    %     Pout = func_ml(M, P, X);
    %     Pout = DOA_IAA(X, A, params);
    %      Pout = CS_OMP(X, A, P);
    %      Pout = DOA_L1Norm(X, A);
    %      Pout = (abs(fft(X,181)))*2/181;

        % 下面是测试深度展开网络的方法
%         [X, T] = ADMM_ANM(X, tau_list, rho_list, eta_list, K);
%         Pout = DOA_MUSIC(T, P, angle_axis);

        RAmap(id, :) = Pout;
    end
end

% save('RAmap.mat','RAmap');

[theta,rho] = meshgrid(angle_axis, range_axis); % 网格化
xaxis = rho .* cosd(theta); % 横坐标
yaxis = rho .* sind(theta); % 纵坐标
surf(yaxis,xaxis,db(abs(RAmap)),'EdgeColor','none');
view(2);colormap('jet');
% caxis([-150 0])
% xlabel('Range(m)','fontsize',15,'fontname','Times New Roman');ylabel('Range(m)','fontsize',15,'fontname','Times New Roman');grid on; axis xy
xlabel('横向距离(m)','fontsize',20,'fontname','宋体');ylabel('纵向距离(m)','fontsize',20,'fontname','宋体');grid on; axis xy
set(gca,'FontSize',20)  %是设置刻度字体大小
set(gca,'GridLineStyle','- -');
set(gca,'GridAlpha',0.2);
set(gca,'LineWidth',1.5);
set(gca,'xminortick','on'); 
set(gca,'ygrid','on','GridColor',[0 0 0]);
set(gca,'looseInset',[0 0 0 0]);
colorbar;

function [PoutMusic] = DOA_MUSIC(RX, P, searchGrids)
    % X: 输入信号 Channel * ChirpNum
    % P: 目标数目
    % PoutMusic: 输出功率谱
    M = size(RX, 1); % 阵元数 RX的时候使用
%     M = size(X, 1); % 阵元数
%     snap = size(X, 2); % 快拍数
%     RX = X * X' / snap; % 协方差矩阵
    
    [V, D] = eig(RX); % 特征值分解
    eig_value = real(diag(D)); % 提取特征值
    [B, I] = sort(eig_value, 'descend'); % 排序特征值
    EN = V(:, I(P+1:end)); % 提取噪声子空间
    
    PoutMusic = zeros(1, length(searchGrids));
    
    for id = 1 : length(searchGrids)
        atheta_vec = exp(-1j * 2 * pi * [0:M-1]' * 1 / 2 * sind(searchGrids(id))); % 导向矢量
        PoutMusic(id) = (abs(1 / (atheta_vec' * EN * EN' * atheta_vec))) ; % 功率谱计算
    end
end