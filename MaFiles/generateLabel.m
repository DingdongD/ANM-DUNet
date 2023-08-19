function Label = generateLabel(thetas)
    % thetas : 真实目标的角度
    P = length(thetas); % 目标数目
    True_Spectrum = 0; % 真实DOA 采用高斯谱累加 依赖于sigmap2
    sigmap2 = 0.5^2; % 高斯核带宽，控制主瓣宽度，值越大宽带越大会影响两个邻近峰值的分辨
    theta_step = 1; % 遍历网格步长
    theta_grids = -90 : theta_step : 90; % 遍历网格区间
    for theta_idx = 1 : P
        True_Spectrum = True_Spectrum + exp(-(theta_grids - thetas(theta_idx)).^2 / 2 / sigmap2);
    end
    Label = True_Spectrum;
end