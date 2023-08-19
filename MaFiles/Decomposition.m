function [PoutMusic] = Decomposition(RX, P, searchGrids)
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
        atheta_vec = exp(1j * 2 * pi * [0:M-1]' * 1 / 2 * sind(searchGrids(id))); % ����ʸ��
        PoutMusic(id) = (abs(1 / (atheta_vec' * EN * EN' * atheta_vec))) ; % �����׼���
    end
end