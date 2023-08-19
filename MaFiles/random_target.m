function thetas = random_target(bound, num, min_dis)
    % seednum : 用于选择最小约束数据还是随机数据
    % bound : 约束目标产生角度区间为[-bound, bound]
    % num : 产生目标数目
    % mindis : 最小距离约束 目标和目标间的距离约束 默认设7
    
    thetas = zeros(1, num);
    
    if num == 1  % bound < 0
        thetas(1, 1) = bound +  randperm(-2*bound, 1);
        return;
    else
        thetas(1, 1) = bound; % [-60,0]随机生成
        for i = 2 : num
            interval = min_dis + randperm(min_dis*2, 1); % （7,21]
            thetas(1,i) = thetas(1, i-1) + interval; 
        end
    end 
end