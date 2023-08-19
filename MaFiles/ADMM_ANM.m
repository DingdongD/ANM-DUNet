function [X_new, T_new] = ADMM_ANM(Y, tau_list, rho_list, eta_list, iter)
    % ��������Զ�����µ�ԭ�ӷ���ANMӦ��ADMM���
    % By Xuliang, 20230304
    % Y: �����ź�
    % tau: �������
    % rho: �ͷ�����
    % iter: ��������
    
    % ��ʼ������ W X u ThetaΪ0
    [M, L] = size(Y); % M ��Ԫ L ����
    
    Lam_old = zeros(M+L, M+L); % [LamT, LamX; LamXH, LamW] 
    The_old = zeros(M+L, M+L); %  LamT-M*M LamW-L*L LamX-M*L
    
    for id = 1 : iter
   
        % ��С�� X W Theta u
        X_new = (Y + Lam_old(1:M,M+1:end) +  Lam_old(M+1:end,1:M)'  + rho_list(id) * The_old(1:M,M+1:end) + rho_list(id) * The_old(M+1:end,1:M)') / (1 + 2 * rho_list(id)); % LamX TheX M*L
        W_new = 1 / rho_list(id) * Lam_old(M+1:end,M+1:end) + The_old(M+1:end,M+1:end) - tau_list(id) / (2*rho_list(id)) *eye(L); % LamW TheW
        
        normalizer = 1 ./ [M;((M-1):-1:1).']; % ��һ��ϵ��
        e1 = zeros(M,1);e1(1)=1;
        u = 1 / rho_list(id) * normalizer .* (toeplitz_adjoint(Lam_old(1:M,1:M))...
                + rho_list(id) * toeplitz_adjoint(The_old(1:M, 1:M)) - tau_list(id) / 2 * M * e1);
%         
%         dummyT = - tau_list(id) / 2 / rho_list(id) * eye(M) + 1 / rho_list(id) * Lam_old(1:M, 1:M) + The_old(1:M, 1:M); % LamT TheT
%         
%         ����T�ĵ�һ��������u_vec
%         u_vec = zeros(M, 1); % �����¶Խ�
%         for iid = 1 : M % ʹ��iȥ����ÿ���Խ���
%             for pid = iid : M
%                 u_vec(iid) = u_vec(iid) + dummyT(pid , pid - iid + 1); % pid ������ iid=3 pid=3:8 pid-iid+1=1:6 
%             end
%             u_vec(iid) = u_vec(iid) / (M - iid + 1);
%         end
%         T_new = toeplitz(u_vec); % �����������Ⱦ���
%         v_vec = zeros(M, 1); % �����϶Խ�
%         for iid = 1 : M
%             for pid = iid : M
%                 v_vec(iid) = v_vec(iid) + dummyT(pid - iid + 1, pid); 
%             end
%             v_vec(iid) = v_vec(iid) / (M - iid + 1);
%         end
        
%         T_new = toeplitz(u_vec, v_vec); % �����������Ⱦ���
        T_new = toeplitz(u);

        The_temp = [T_new, X_new; X_new', W_new] - 1 / rho_list(id) * Lam_old; % �����������ֵ�ֽ�
        [The_G, The_D] = eig(The_temp);
        diag_data = diag(The_D);
        data_idx = find(diag_data>0);
        The_new = The_G(:, data_idx) * diag(diag_data(data_idx)) * pinv(The_G(:, data_idx)); 
        The_new = (The_new + The_new') / 2;
        Lam_new = Lam_old + eta_list(id) * (The_new - [T_new, X_new; X_new', W_new]);
        
        % ����
        Lam_old = Lam_new;
        The_old = The_new;
        
    end
    