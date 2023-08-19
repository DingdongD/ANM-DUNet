%% ���ļ��������ɻ��������ݼ�
clc;clear;close all;

%% ������ʼ��
%% ����ģ�Ͳ���
M = 8; % ��Ԫ��Ŀ
a = [0:M-1]'; % �������
f0 = 77e9; % Ƶ��
c = 3e8; % ����
lambda = c / f0; % ����
d = lambda / 2; % ��Ԫ���

%% ��ʼ��Ŀ�����ɵĲ���
maxTarget = 3; % ���Ŀ����Ŀ
% snr_db = 0 : 1 : 35; % ���������
snaps = 0 : 2 : 100; % ��������
maxSample = 300; % ���Ŀ��������
lower_bound = 60; % ��СĿ�����ɱ߽� ����FOV����
mindis = 7; % �ڽ�Ŀ��Լ�����

%% �ź�ģ�Ͳ���
fs = 1000; % ����Ƶ��
% t = 1 / fs * (0 : snap - 1); % ʱ��
% noise_flag = 1; % ������־
theta_step = 1; % ���񲽳�
theta_grids = -90 : theta_step : 90; % ������������
thetaNum = length(theta_grids); % ������Ŀ

%% �ļ��������
% root_dir = '..\NoiselessDatafiles\'; % NoiselessDatafiles��δ����汾�� NoiselessSets������汾��
root_dir = '..\Snap100File\'; % NoiselessDatafiles��δ����汾�� NoiselessSets������汾��
extended_flag = true;  % ������

%% ��������
for target_id = 1 : maxTarget
    for snap_id = 51 : 51
        P = target_id; % ��Դ��Ŀ
        thetas = zeros(1, P); % ��ʼ��Ŀ��洢���� ������*Ŀ����
        disp(['���ڴ����ļ�----->TAR/SNAP:',num2str(target_id),'/',num2str(snaps(snap_id))])
        tic
        for sample_id = 1 : maxSample
            S = (randn(P, snaps(snap_id)) + 1j * randn(P, snaps(snap_id))) / sqrt(2); % �ź�
     
            bound = -lower_bound + randperm(lower_bound,1) - 1; % �߽�Լ��[-60,0)
            thetas(1,:) = random_target(bound, P, mindis); % ������ʵĿ��Ƕ�
            theta_vec = exp(-1j * 2 * pi * d / lambda * a * sind(thetas(1, :))); % ����ʸ��
            Y = theta_vec * S; % ��������ź�
            
            [PoutANM,u_vec] = DOA_ANM(Y, P);
            PoutANM = round(PoutANM);
            thetas = sort(thetas);
            PoutANM = sort(PoutANM);
            err = sum(abs(PoutANM - thetas));
            while err > 1
                thetas(1,:) = random_target(bound, P, mindis); % ������ʵĿ��Ƕ�
                theta_vec = exp(-1j * 2 * pi * d / lambda * a * sind(thetas(1, :))); % ����ʸ��
                X0 = theta_vec * S; % ��������ź�
                Y = X0;
                [PoutANM,u_vec] = DOA_ANM(Y, P);
                PoutANM = round(PoutANM);
                thetas = sort(thetas);
                PoutANM = sort(PoutANM);
                err = sum(abs(PoutANM - thetas));
            end
            
%             X0 = theta_vec * S; % ��������ź�
%             Y = awgn(X0, snr_db(snr_id), 'measured');
%             Y = X0;

%             if extended_flag
%                 temp_data = [Y, repmat(Y(:,end),1,max(snaps)-snaps(snap_id))];
%                 Y = temp_data;
%             end

            % ������ʵĿ��ı�ǩ[α��˹��]
            Ylabel = u_vec;
%             Ylabel = generateLabel(thetas); % 1 * P  % ���ֱ�ǩ�ܺ�ʱ��
            % �������ݵ�ָ���ļ�������
            Filename = strcat(root_dir, 'Tar', num2str(target_id),'_',...
                        '_','Snap',num2str(snaps(snap_id)), '\');

            Datafilename = strcat(Filename, 'Data\');
            Labelfilename = strcat(Filename,'Label\');
            if ~exist(Datafilename,'dir') || ~exist(Labelfilename,'dir')  % ����Ƿ��и�Ŀ¼�ļ� ���򴴽�
                mkdir(Datafilename);
                mkdir(Labelfilename);
            end

            matfile = strcat(Datafilename,'SampleId', num2str(sample_id),'.mat'); % ���￼�ǻ��ǵ�������ȽϺ� ��Ϊ����ά�ȱȽ���
            labelfile = strcat(Labelfilename,'LabelId', num2str(sample_id),'.mat');  % ��ǩ�ļ�����
            save(matfile, 'Y');   
            save(labelfile,'Ylabel'); 
        end
    toc
    end
end

%% ���ӻ�
figure(1);
plot(theta_grids, Ylabel); xlabel('Angle(deg)');ylabel('Amplitude');
