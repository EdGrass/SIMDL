function [ p ] = iwf( P_total, sigma2, H, K )
    % 输入参数验证
    if nargin ~= 4
        error('需要4个输入参数: P_total, sigma2, H, K');
    end
    if P_total <= 0 || sigma2 <= 0
        error('P_total 和 sigma2 必须为正数');
    end
    if size(H,1) ~= K || size(H,2) ~= K
        error('H矩阵维度必须为 KxK');
    end

    gama = zeros(K, 1);
    R = zeros(K, 1);
    p = P_total/K*ones(K, 1);
    p_new = zeros(K, 1);
    water_floor = zeros(K, 1);
    noise = zeros(K, 1);
    for ii = 1 : K
        noise(ii) = abs(H(ii, :)).^2*p - abs(H(ii, ii))^2*p(ii) + sigma2;
        gama(ii) = (abs(H(ii, ii))^2*p(ii))/noise(ii);
        R(ii) = log2(1 + gama(ii));
    end
    sum_R_old = sum(R);
    sum_R_new = 2*sum_R_old;
    mm = 1;
    while abs(sum_R_new - sum_R_old) >= sum_R_old*0.000001 && mm <= 100
        sum_R_old = sum_R_new;
        for ii = 1 : K
            water_floor(ii) = noise(ii)/abs(H(ii,ii))^2;
        end
        [water_floor, order] = sort(water_floor);
        P_sum = 0;
        for ii = 1 : K - 1
            P_sum = P_sum + ii*(water_floor(ii+1) - water_floor(ii)); % Calculate the sum power when all (1:ii) channels are filled
            if P_sum >= P_total
                break
            end
        end
        if P_sum >= P_total
            PA_WF_part = water_floor(ii+1) - water_floor(1:ii) - (P_sum - P_total)/ii; % Subtract the excess power for all (1:ii) channels
            PA_WF = [PA_WF_part; zeros(K - length(PA_WF_part), 1)]; % Match the dimension
        else
            PA_WF = water_floor(ii+1) - water_floor(1:(ii+1)) + (P_total - P_sum)/(ii+1); % Add the extra power for all (1:ii+1) channels
        end
        p_new(order) = PA_WF;
        p = 1/K*p_new + (K-1)/K*p;
        for ii = 1:K
            noise(ii) = abs(H(ii, :)).^2*p - abs(H(ii, ii))^2*p(ii) + sigma2;
            gama(ii) = (abs(H(ii, ii))^2*p(ii))/noise(ii);
            R(ii) = log2(1 + gama(ii));
        end
        sum_R_new = sum(R);
        mm = mm+1;
    end

    % 添加收敛性检查
    if mm >= 100
        warning('IWF算法达到最大迭代次数但可能未收敛');
    end

    % 添加结果验证
    if any(isnan(p)) || any(p < 0)
        error('功率分配结果无效');
    end
end