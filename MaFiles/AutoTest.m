%% ���ļ����ڴ���������-���غ��ײ��״���Ĳ�������
%% By Xuliang, 20230408

clc;clear;close all;
arr_path = 'H:\ADMM-ANM\Mafiles\AutoData\frame2\'; % �����Ų���·��
data_path = 'H:\ADMM-ANM\ProcessedDataset\mode2\'; % �������ݵ�·��

array_mat = strcat(arr_path,'array.mat');
rangefft_mat = strcat(data_path, 'rfft4.mat'); 

rangefft_data = load(rangefft_mat); % ��ȡ����άFFT����
rangefft_data = rangefft_data.data;
disp(['�����ߴ�Ϊ��',num2str(size(rangefft_data))]); 
% mode1: 256 * 128(������100m������ֱ���0.39m������ٶ�4.55m/s���ٶȷֱ���0.07m/s)
% mode2: 256 * 128(������50m������ֱ���0.2m������ٶ�6.28m/s,�ٶȷֱ���0.06m/s)
% mode3: 128 * 255(������50m������ֱ���0.39m������ٶ�6.28m/s���ٶȷֱ���0.05m/s)
% mode4: 512 * 64(������50m������ֱ���0.1m������ٶ�2.39m/s���ٶȷֱ���0.07m/s)

array_data = load(array_mat); % ��ȡ��Ԫ��������
array_data = array_data.data; 

% ��ȡ��λά����
range_data = zeros(size(rangefft_data,1),size(rangefft_data,2), size(array_data,1));
for i = 1 : size(array_data,1)
    range_data(:, :, i) = squeeze(rangefft_data(:, :,array_data(i,1)+1, array_data(i,2)+1));
end
disp(['���������ߴ�Ϊ��',num2str(size(range_data))]); % 256 * 128 * 16 * 12

mode = 2;

angle_axis = [-90:90]; % �Ƕ�����
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


% ���BEV
for id = 1:size(range_data,1)
    X = squeeze(range_data(id, :, :)); % ��ȡ�����Ļ����� 
%     X = sum(X,1); % ��̬���������ö�����ۻ� ��̬�������ʺ�
    
    if size(X,1) ~= 86 
        X = X.'; % �����ת��
    end

    % ����FFT����[����Ĳ�����FFT]
%     Pout = (abs(fft(X,181)))*2/181;
    
%     Pout = DOA_MUSIC(X, P, angle_axis);
%     % �����ǲ������չ������ķ���
%     [X, T] = ADMM_ANM(X, tau_list, rho_list, eta_list, K);
%     Pout =Decomposition(T, P, angle_axis);

    T = DOA_ANM(X, P);
    Pout =Decomposition(T, P, angle_axis);
    
    RAmap(id, :) = Pout;
end

[theta,rho] = meshgrid(angle_axis, range_axis); % ����
xaxis = rho .* cosd(theta); % ������
yaxis = rho .* sind(theta); % ������
figure(1);
surf(yaxis,xaxis,db(abs(RAmap)),'EdgeColor','none');
view(2);colormap('jet');
ylim([0 max_range]);xlim([-max_range max_range]);
% caxis([-40 -30])
% xlabel('Range(m)','fontsize',15,'fontname','Times New Roman');ylabel('Range(m)','fontsize',15,'fontname','Times New Roman');grid on; axis xy
xlabel('�������(m)','fontsize',20,'fontname','����');ylabel('�������(m)','fontsize',20,'fontname','����');grid on; axis xy
set(gca,'FontSize',20)  %�����ÿ̶������С
set(gca,'GridLineStyle','- -');
set(gca,'GridAlpha',0.2);
set(gca,'LineWidth',1.5);
set(gca,'xminortick','on'); 
set(gca,'ygrid','on','GridColor',[0 0 0]);
set(gca,'looseInset',[0 0 0 0]);
colorbar;
