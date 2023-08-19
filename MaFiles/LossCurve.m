clc;clear;close all;

%% 本文件用于绘制LOSS曲线 

train_mat_path = 'H:\ADMM-ANM\Pyfiles\snap5_logs1\train_loss.mat';
valid_mat_path = 'H:\ADMM-ANM\Pyfiles\snap5_logs1\val_loss.mat';
train_mat_file = load(train_mat_path);
valid_mat_file = load(valid_mat_path);

train_loss = train_mat_file.Y.train_loss;
val_loss = valid_mat_file.Y.val_loss;

%% 可视化分析
figure(1);
plot(1:length(train_loss),train_loss, 'r*:'); hold on;
plot(1:length(val_loss),val_loss, 'b.:'); hold on;
xlabel('Iter Num'); ylabel('Loss');

