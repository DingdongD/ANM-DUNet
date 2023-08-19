%% ���ļ����ڲ��Զ���ʵ�⾲̬���س����µ�RAͼ


clc;clear;close all;
f0 = 77e9; % Ƶ��
c = 3e8; % ����
lambda = c / f0; % ����
d = lambda / 2; % ��Ԫ���

range_data = load('range_fft.mat');
range_data = range_data.range_fft;

angle_axis = [-90:90]; % �Ƕ�����
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
A = exp(-1j * 2 * pi * d / lambda * a * sind(angle_axis)); % ������걸�ֵ�
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
        [V, D] = eig(RX); % ����ֵ�ֽ�
        figure(1); % ����ֵ�ֲ��Ŀ��ӻ�
        
        SP = V(:, M-P+1:M);
        EN = V(:, 1:M-P);
        s_eigen_fft = fft(SP);
        n_eigen_fft = fft(EN);
        plot(1:M,abs(s_eigen_fft), 'ks-','LineWidth',1.5); hold on;
        plot(1:M,abs(n_eigen_fft), 'r','LineWidth',1);
        legend('Signal Space', 'Noise Space','location','NorthWest');

%         plot(diag((D)),'kd-');
        xlabel('Number of Eigenvalues'); ylabel('Eigenvalues');
        title(['Ŀ����ţ�',num2str(id)]);
        
        drawnow; % ˢ����Ļ
        F = getframe(gcf); % getframe������������ͼ����ΪӰƬ֡��gcf���ص�ǰfigure�ľ��
        I = frame2im(F); % frame2im������ӰƬ֡������ͼ�����ݡ����һ����ά����
        [I,map]=rgb2ind(I,256);    %rgb2ind��rgbͼ��ת��������(index)ͼ��
        if pic_num == 1
            imwrite(I,map,'test1.gif','gif', 'Loopcount',inf,'DelayTime',0.1);
        else
            imwrite(I,map,'test1.gif','gif','WriteMode','append','DelayTime',0.1);
        end
        pic_num = pic_num + 1;
        
        pause(0.1);
        hold off
    else
        % ������MUSIC�ķ���
        Pout = DOA_MUSIC(RX, 2, angle_axis);
    %     Pout = func_ml(M, P, X);
    %     Pout = DOA_IAA(X, A, params);
    %      Pout = CS_OMP(X, A, P);
    %      Pout = DOA_L1Norm(X, A);
    %      Pout = (abs(fft(X,181)))*2/181;

        % �����ǲ������չ������ķ���
%         [X, T] = ADMM_ANM(X, tau_list, rho_list, eta_list, K);
%         Pout = DOA_MUSIC(T, P, angle_axis);

        RAmap(id, :) = Pout;
    end
end

% save('RAmap.mat','RAmap');

[theta,rho] = meshgrid(angle_axis, range_axis); % ����
xaxis = rho .* cosd(theta); % ������
yaxis = rho .* sind(theta); % ������
surf(yaxis,xaxis,db(abs(RAmap)),'EdgeColor','none');
view(2);colormap('jet');
% caxis([-150 0])
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

function [PoutMusic] = DOA_MUSIC(RX, P, searchGrids)
    % X: �����ź� Channel * ChirpNum
    % P: Ŀ����Ŀ
    % PoutMusic: ���������
    M = size(RX, 1); % ��Ԫ�� RX��ʱ��ʹ��
%     M = size(X, 1); % ��Ԫ��
%     snap = size(X, 2); % ������
%     RX = X * X' / snap; % Э�������
    
    [V, D] = eig(RX); % ����ֵ�ֽ�
    eig_value = real(diag(D)); % ��ȡ����ֵ
    [B, I] = sort(eig_value, 'descend'); % ��������ֵ
    EN = V(:, I(P+1:end)); % ��ȡ�����ӿռ�
    
    PoutMusic = zeros(1, length(searchGrids));
    
    for id = 1 : length(searchGrids)
        atheta_vec = exp(-1j * 2 * pi * [0:M-1]' * 1 / 2 * sind(searchGrids(id))); % ����ʸ��
        PoutMusic(id) = (abs(1 / (atheta_vec' * EN * EN' * atheta_vec))) ; % �����׼���
    end
end