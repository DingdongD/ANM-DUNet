%% 本文件用于处理本课题组-车载毫米波雷达组的部分数据
%% By Xuliang, 20230408

clc;clear;close all;
arr_path = 'H:\ADMM-ANM\Mafiles\AutoData\frame2\'; % 阵列排布的路径
data_path = 'H:\ADMM-ANM\ProcessedDataset\mode2\'; % 处理数据的路径

array_mat = strcat(arr_path,'array.mat');
rangefft_mat = strcat(data_path, 'rfft4.mat'); 

rangefft_data = load(rangefft_mat); % 读取距离维FFT数据
rangefft_data = rangefft_data.data;
disp(['张量尺寸为：',num2str(size(rangefft_data))]); 
% mode1: 256 * 128(最大距离100m，距离分辨率0.39m，最大速度4.55m/s，速度分辨率0.07m/s)
% mode2: 256 * 128(最大距离50m，距离分辨率0.2m，最大速度6.28m/s,速度分辨率0.06m/s)
% mode3: 128 * 255(最大距离50m，距离分辨率0.39m，最大速度6.28m/s，速度分辨率0.05m/s)
% mode4: 512 * 64(最大距离50m，距离分辨率0.1m，最大速度2.39m/s，速度分辨率0.07m/s)

array_data = load(array_mat); % 读取阵元排列数据
array_data = array_data.data; 

% 获取方位维数据
range_data = zeros(size(rangefft_data,1),size(rangefft_data,2), size(array_data,1));
for i = 1 : size(array_data,1)
    range_data(:, :, i) = squeeze(rangefft_data(:, :,array_data(i,1)+1, array_data(i,2)+1));
end
disp(['重排张量尺寸为：',num2str(size(range_data))]); % 256 * 128 * 16 * 12

mode = 2;

angle_axis = [-90:90]; % 角度索引
if mode == 1
    max_range = 100.06;
    range_axis = [0:max_range/(size(range_data,1)-1):max_range];
elseif mode == 2
    max_range = 50;
    range_axis = [0:max_range/(size(range_data,1)-1):max_range];
elseif mode == 3
    max_range = 50;
    range_axis = [0:max_range/(size(range_data,1)-1):max_range];
elseif mode == 4
    max_range = 50;
    range_axis = [0:max_range/(size(range_data,1)-1):max_range];
end
    
P = 3;
RAmap = zeros(size(range_data,1), length(angle_axis)); % 256*128

para_path = 'H:\ADMM-ANM\Pyfiles\snap5_logs1\para.mat';
para_list = load(para_path);
para_data = para_list.Para;
K = length(para_data)/3;

rho_list = para_data(1:K);
tau_list = para_data(K+1:2*K);
eta_list = para_data(2*K+1:3*K);


% 输出BEV
for id = 1:size(range_data,1)
    X = squeeze(range_data(id, :, :)); % 提取单快拍或多快拍 
%     X = sum(X,1); % 静态场景可以用多快拍累积 动态场景不适合
    
    if size(X,1) ~= 86 
        X = X.'; % 多快拍转置
    end

    % 采用FFT方法[多快拍不考虑FFT]
%     Pout = (abs(fft(X,181)))*2/181;
    
%     Pout = DOA_MUSIC(X, P, angle_axis);
%     % 下面是测试深度展开网络的方法
%     [X, T] = ADMM_ANM(X, tau_list, rho_list, eta_list, K);
%     Pout =Decomposition(T, P, angle_axis);

    T = DOA_ANM(X, P);
    Pout =Decomposition(T, P, angle_axis);
    
    RAmap(id, :) = Pout;
end

[theta,rho] = meshgrid(angle_axis, range_axis); % 网格化
xaxis = rho .* cosd(theta); % 横坐标
yaxis = rho .* sind(theta); % 纵坐标
figure(1);
surf(yaxis,xaxis,db(abs(RAmap)),'EdgeColor','none');
view(2);colormap('jet');
ylim([0 max_range]);xlim([-max_range max_range]);
% caxis([-40 -30])
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
