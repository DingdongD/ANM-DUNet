%% 本文件用于生成基础版数据集
clc;clear;close all;

%% 参数初始化
%% 阵列模型参数
M = 8; % 阵元数目
a = [0:M-1]'; % 导向序号
f0 = 77e9; % 频率
c = 3e8; % 光速
lambda = c / f0; % 波长
d = lambda / 2; % 阵元间距

%% 初始化目标生成的参数
maxTarget = 3; % 最大目标数目
% snr_db = 0 : 1 : 35; % 信噪比区间
snaps = 0 : 2 : 100; % 快拍区间
maxSample = 300; % 最大目标样本数
lower_bound = 60; % 最小目标生成边界 根据FOV衡量
mindis = 7; % 邻近目标约束间隔

%% 信号模型参数
fs = 1000; % 采样频率
% t = 1 / fs * (0 : snap - 1); % 时间
% noise_flag = 1; % 噪声标志
theta_step = 1; % 网格步长
theta_grids = -90 : theta_step : 90; % 遍历网格区间
thetaNum = length(theta_grids); % 网格数目

%% 文件保存参数
% root_dir = '..\NoiselessDatafiles\'; % NoiselessDatafiles是未扩充版本的 NoiselessSets是扩充版本的
root_dir = '..\Snap100File\'; % NoiselessDatafiles是未扩充版本的 NoiselessSets是扩充版本的
extended_flag = true;  % 扩充标记

%% 数据生成
for target_id = 1 : maxTarget
    for snap_id = 51 : 51
        P = target_id; % 信源数目
        thetas = zeros(1, P); % 初始化目标存储矩阵 样本数*目标数
        disp(['正在处理文件----->TAR/SNAP:',num2str(target_id),'/',num2str(snaps(snap_id))])
        tic
        for sample_id = 1 : maxSample
            S = (randn(P, snaps(snap_id)) + 1j * randn(P, snaps(snap_id))) / sqrt(2); % 信号
     
            bound = -lower_bound + randperm(lower_bound,1) - 1; % 边界约束[-60,0)
            thetas(1,:) = random_target(bound, P, mindis); % 生成真实目标角度
            theta_vec = exp(-1j * 2 * pi * d / lambda * a * sind(thetas(1, :))); % 导向矢量
            Y = theta_vec * S; % 无噪基带信号
            
            [PoutANM,u_vec] = DOA_ANM(Y, P);
            PoutANM = round(PoutANM);
            thetas = sort(thetas);
            PoutANM = sort(PoutANM);
            err = sum(abs(PoutANM - thetas));
            while err > 1
                thetas(1,:) = random_target(bound, P, mindis); % 生成真实目标角度
                theta_vec = exp(-1j * 2 * pi * d / lambda * a * sind(thetas(1, :))); % 导向矢量
                X0 = theta_vec * S; % 无噪基带信号
                Y = X0;
                [PoutANM,u_vec] = DOA_ANM(Y, P);
                PoutANM = round(PoutANM);
                thetas = sort(thetas);
                PoutANM = sort(PoutANM);
                err = sum(abs(PoutANM - thetas));
            end
            
%             X0 = theta_vec * S; % 无噪基带信号
%             Y = awgn(X0, snr_db(snr_id), 'measured');
%             Y = X0;

%             if extended_flag
%                 temp_data = [Y, repmat(Y(:,end),1,max(snaps)-snaps(snap_id))];
%                 Y = temp_data;
%             end

            % 产生真实目标的标签[伪高斯谱]
            Ylabel = u_vec;
%             Ylabel = generateLabel(thetas); % 1 * P  % 这种标签很耗时间
            % 保存数据到指定文件夹下面
            Filename = strcat(root_dir, 'Tar', num2str(target_id),'_',...
                        '_','Snap',num2str(snaps(snap_id)), '\');

            Datafilename = strcat(Filename, 'Data\');
            Labelfilename = strcat(Filename,'Label\');
            if ~exist(Datafilename,'dir') || ~exist(Labelfilename,'dir')  % 检查是否有该目录文件 无则创建
                mkdir(Datafilename);
                mkdir(Labelfilename);
            end

            matfile = strcat(Datafilename,'SampleId', num2str(sample_id),'.mat'); % 这里考虑还是单独保存比较好 因为后面维度比较乱
            labelfile = strcat(Labelfilename,'LabelId', num2str(sample_id),'.mat');  % 标签文件保存
            save(matfile, 'Y');   
            save(labelfile,'Ylabel'); 
        end
    toc
    end
end

%% 可视化
figure(1);
plot(theta_grids, Ylabel); xlabel('Angle(deg)');ylabel('Amplitude');
