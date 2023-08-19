function [X_new, T_new] = ADMM_ANM(Y, tau_list, rho_list, eta_list, iter)
    % 本程序针对多快拍下的原子范数ANM应用ADMM求解
    % By Xuliang, 20230304
    % Y: 基带信号
    % tau: 正则参数
    % rho: 惩罚因子
    % iter: 迭代次数
    
    % 初始化变量 W X u Theta为0
    [M, L] = size(Y); % M 阵元 L 快拍
    
    Lam_old = zeros(M+L, M+L); % [LamT, LamX; LamXH, LamW] 
    The_old = zeros(M+L, M+L); %  LamT-M*M LamW-L*L LamX-M*L
    
    for id = 1 : iter
   
        % 最小化 X W Theta u
        X_new = (Y + Lam_old(1:M,M+1:end) +  Lam_old(M+1:end,1:M)'  + rho_list(id) * The_old(1:M,M+1:end) + rho_list(id) * The_old(M+1:end,1:M)') / (1 + 2 * rho_list(id)); % LamX TheX M*L
        W_new = 1 / rho_list(id) * Lam_old(M+1:end,M+1:end) + The_old(M+1:end,M+1:end) - tau_list(id) / (2*rho_list(id)) *eye(L); % LamW TheW
        
        normalizer = 1 ./ [M;((M-1):-1:1).']; % 归一化系数
        e1 = zeros(M,1);e1(1)=1;
        u = 1 / rho_list(id) * normalizer .* (toeplitz_adjoint(Lam_old(1:M,1:M))...
                + rho_list(id) * toeplitz_adjoint(The_old(1:M, 1:M)) - tau_list(id) / 2 * M * e1);
%         
%         dummyT = - tau_list(id) / 2 / rho_list(id) * eye(M) + 1 / rho_list(id) * Lam_old(1:M, 1:M) + The_old(1:M, 1:M); % LamT TheT
%         
%         构造T的第一列列向量u_vec
%         u_vec = zeros(M, 1); % 构建下对角
%         for iid = 1 : M % 使用i去控制每个对角线
%             for pid = iid : M
%                 u_vec(iid) = u_vec(iid) + dummyT(pid , pid - iid + 1); % pid 控制列 iid=3 pid=3:8 pid-iid+1=1:6 
%             end
%             u_vec(iid) = u_vec(iid) / (M - iid + 1);
%         end
%         T_new = toeplitz(u_vec); % 构建托普利兹矩阵
%         v_vec = zeros(M, 1); % 构建上对角
%         for iid = 1 : M
%             for pid = iid : M
%                 v_vec(iid) = v_vec(iid) + dummyT(pid - iid + 1, pid); 
%             end
%             v_vec(iid) = v_vec(iid) / (M - iid + 1);
%         end
        
%         T_new = toeplitz(u_vec, v_vec); % 构建托普利兹矩阵
        T_new = toeplitz(u);

        The_temp = [T_new, X_new; X_new', W_new] - 1 / rho_list(id) * Lam_old; % 对其进行特征值分解
        [The_G, The_D] = eig(The_temp);
        diag_data = diag(The_D);
        data_idx = find(diag_data>0);
        The_new = The_G(:, data_idx) * diag(diag_data(data_idx)) * pinv(The_G(:, data_idx)); 
        The_new = (The_new + The_new') / 2;
        Lam_new = Lam_old + eta_list(id) * (The_new - [T_new, X_new; X_new', W_new]);
        
        % 更新
        Lam_old = Lam_new;
        The_old = The_new;
        
    end
    