function [PoutMusic] = Decomposition(RX, P, searchGrids)
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
        atheta_vec = exp(1j * 2 * pi * [0:M-1]' * 1 / 2 * sind(searchGrids(id))); % 导向矢量
        PoutMusic(id) = (abs(1 / (atheta_vec' * EN * EN' * atheta_vec))) ; % 功率谱计算
    end
end