function Label = generateLabel(thetas)
    % thetas : ��ʵĿ��ĽǶ�
    P = length(thetas); % Ŀ����Ŀ
    True_Spectrum = 0; % ��ʵDOA ���ø�˹���ۼ� ������sigmap2
    sigmap2 = 0.5^2; % ��˹�˴������������ȣ�ֵԽ����Խ���Ӱ�������ڽ���ֵ�ķֱ�
    theta_step = 1; % �������񲽳�
    theta_grids = -90 : theta_step : 90; % ������������
    for theta_idx = 1 : P
        True_Spectrum = True_Spectrum + exp(-(theta_grids - thetas(theta_idx)).^2 / 2 / sigmap2);
    end
    Label = True_Spectrum;
end