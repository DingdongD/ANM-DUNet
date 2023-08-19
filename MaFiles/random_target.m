function thetas = random_target(bound, num, min_dis)
    % seednum : ����ѡ����СԼ�����ݻ����������
    % bound : Լ��Ŀ������Ƕ�����Ϊ[-bound, bound]
    % num : ����Ŀ����Ŀ
    % mindis : ��С����Լ�� Ŀ���Ŀ���ľ���Լ�� Ĭ����7
    
    thetas = zeros(1, num);
    
    if num == 1  % bound < 0
        thetas(1, 1) = bound +  randperm(-2*bound, 1);
        return;
    else
        thetas(1, 1) = bound; % [-60,0]�������
        for i = 2 : num
            interval = min_dis + randperm(min_dis*2, 1); % ��7,21]
            thetas(1,i) = thetas(1, i-1) + interval; 
        end
    end 
end