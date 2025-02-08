clc;
clearvars;
close all;

% 添加在文件开头
try
    % 检查MATLAB版本和工具箱
    ver_info = ver;
    required_toolboxes = {'MATLAB', 'Signal Processing Toolbox'};
    
    for i = 1:length(required_toolboxes)
        if ~any(strcmp({ver_info.Name}, required_toolboxes{i}))
            error('缺少必要的工具箱: %s', required_toolboxes{i});
        end
    end
    
    % 替换内存检查为更通用的方式
    try
        % 尝试创建一个小的测试数组来检查内存可用性
        test_size = 1000;
        test_array = zeros(test_size, test_size);
        clear test_array;
        fprintf('内存检查通过\n');
    catch
        warning('内存可能不足，建议关闭其他应用程序');
    end
catch e
    fprintf('初始化检查失败: %s\n', e.message);
    return;
end

% 添加函数路径
addpath(pwd);  % 添加当前目录到搜索路径

% 修改为正常训练规模
DEBUG_MODE = false;  % 关闭测试模式

% 设置正常参数
MonteCarlo = 10;     % 恢复原始Monte Carlo次数
Layer_Max = 10;      % 恢复原始层数
K_values = 2:8;      % 恢复原始用户数量范围
N = 100;             % 恢复原始超原子数量

sigma2 = 10^(-104/10);
c = 3*10^8;
f0 = 28*10^9;
lambda = c/f0;
d_element_SIM = lambda/2;
MonteCarlo = 10;
n_max = 10;
N = 100;
K = 4;
Layer_Max = 10;
Pt = 10^(10/10)*10^(5/10);
R_MonteCarlo = zeros(MonteCarlo,1);
W_T = zeros(N, N);
Corr_T = zeros(N, N);
gama = zeros(K, 1);
R = zeros(K, 1);
location = [10*(1:K).' 10*(1:K).'];
W_T_1 = zeros(N, K);
R_layer = zeros(Layer_Max, 1);
gradient_temp = zeros(K, K);
del = zeros(K, 1);

% 在主循环开始前添加
fprintf('开始优化过程...\n');
fprintf('总共需要处理 %d 层\n', Layer_Max);

for cc = 1 : Layer_Max
    fprintf('\n处理第 %d 层:\n', cc);
    L = cc;
    phase_transmit=zeros(N,L);
    d_layer = 5*lambda/L;
    for mm1 = 1 : N
        m_z = ceil(mm1/n_max);
        m_x = mod(mm1-1,n_max)+1;
        for mm2 = 1 : N
            n_z = ceil(mm2/n_max);
            n_x = mod(mm2-1,n_max)+1;
            d_temp  = sqrt( (m_x-n_x)^2 +  (m_z-n_z) ^2 )*d_element_SIM;
            d_temp2 = sqrt(d_layer^2 + d_temp^2);
            W_T(mm2,mm1) = lambda^2/4*(d_layer/d_temp2/d_temp2*(1/2/pi/d_temp2-1i/lambda))*exp(1i*2*pi*d_temp2/lambda);
            Corr_T(mm2,mm1) = sinc(2*d_temp/lambda);
        end
    end
    for mm = 1:N
        m_y = ceil(mm/n_max);
        m_x = mod(mm-1,n_max)+1;
        for nn = 1:K
            d_transmit = sqrt(d_layer^2 + ...
                ( (m_x-(1+n_max)/2)*d_element_SIM - (nn-(1+K)/2)*lambda/2)^2 + ...
                ( (m_y-(1+n_max)/2)*d_element_SIM )^2 );
            W_T_1(mm,nn) = lambda^2/4*(d_layer/d_transmit/d_transmit*(1/2/pi/d_transmit-1i/lambda))*exp(1i*2*pi/d_transmit/lambda);
        end
    end
    
    tic  % 开始计时
    fprintf('正在进行Monte Carlo迭代 (%d 次)...\n', MonteCarlo);
    
    PD_transmit_phase = zeros(N,L);
    d = sqrt((10-5*lambda)^2 + location(:,1).^2+location(:,2).^2);
    pathloss = (lambda/(4*pi))^2./d.^(3.5);
    
    % 简化数据存储结构
    all_G = {};
    all_phase_transmit = {};
    all_R = {};
    all_pathloss = {};
    all_location = {};
    
    for jj = 1:MonteCarlo
	    rng(jj)
        fprintf('Monte Carlo迭代 %d/%d\n', jj, MonteCarlo);
        G_in = sqrt(1/2)*(randn(K,N)+1i*randn(K,N));
        G = diag(sqrt(pathloss))*G_in*(Corr_T)^(1/2);
        phase_transmit = randn(N,L) + 1i*randn(N,L);
        phase_transmit = phase_transmit./abs(phase_transmit);
        W_SIM = diag(phase_transmit(:,1))*W_T_1;
        for l=1:L-1
            W_SIM = diag(phase_transmit(:,l+1))*W_T*W_SIM;
        end
        H_fit = G*W_SIM;
        p = iwf( Pt, sigma2, H_fit, K );
        for ii = 1:K
            gama(ii) = (abs(H_fit(ii, ii))^2*p(ii))/(abs(H_fit(ii, :)).^2*p - abs(H_fit(ii, ii))^2*p(ii) + sigma2);
            R(ii) = log2(1+gama(ii));
        end
        C_old = sum(R);
        C_new = C_old*2;
        phase_phase_transmit = angle(phase_transmit);
        count = 1;
        while abs(C_new-C_old) >= C_old * 0.000001 && count <= 50
            C_old = C_new;
            for ll = 1:L
                for mm = 1:N
                    X_left = W_T_1;
                    for ll_left = 1:ll-1
                        X_left = W_T*diag(phase_transmit(:,ll_left))*X_left;
                    end
                    X_right = G;
                    for ll_right = 1:(L-ll)
                        X_right = X_right*diag(phase_transmit(:,L+1-ll_right))*W_T;
                    end
                    for ss1 = 1:K
                        del(ss1) = 1/(abs(H_fit(ss1,:)).^2*p+sigma2);
                        for ss2 = 1:K
                            temp1 = X_right(ss1,mm)*X_left(mm,ss2);
                            if ss2 == ss1
                                gradient_temp(ss1,ss2) = del(ss1)*p(ss1)*imag((phase_transmit(mm,ll)*temp1)'*(H_fit(ss1,ss2)));
                            else
                                gradient_temp(ss1,ss2) = -del(ss1)*gama(ss1)*p(ss2)*imag((phase_transmit(mm,ll)*temp1)'*(H_fit(ss1,ss2)));
                            end
                        end
                    end
                    PD_transmit_phase(mm,ll) = 2/log(2)*sum(sum(gradient_temp));
                end
            end
            yy = pi/max(max(PD_transmit_phase));
            C_old_1 = C_old;
            C_new = 0;
            count_2 = 1;
            phase_transmit_temp = phase_transmit;
            while C_new < C_old_1 && count_2 <= 20
                phase_phase_transmit_temp = phase_phase_transmit+yy*PD_transmit_phase;
                phase_transmit_temp = exp(1i*phase_phase_transmit_temp);
                W_SIM = diag(phase_transmit_temp(:,1))*W_T_1;
                for l=1:L-1
                    W_SIM = diag(phase_transmit_temp(:,l+1))*W_T*W_SIM;
                end
                H_fit = G*W_SIM;
                p = iwf( Pt, sigma2, H_fit, K );
                for ii = 1:K
                    gama(ii) = (abs(H_fit(ii,ii))^2*p(ii))/(abs(H_fit(ii,:)).^2*p - abs(H_fit(ii,ii))^2*p(ii) + sigma2);
                    R(ii) = log2(1+gama(ii));
                end
                C_new = sum(R);
                yy = yy*0.5;
                count_2 = count_2+1;
            end
            phase_transmit = phase_transmit_temp;
            phase_phase_transmit = angle(phase_transmit);
            count = count+ 1;
            
            % 使用cell数组保存数据
            all_G{end+1} = G;
            all_phase_transmit{end+1} = phase_transmit;
            all_R{end+1} = R;
            all_pathloss{end+1} = pathloss;
            all_location{end+1} = location;
        end
        R_MonteCarlo(jj) = C_new;
        fprintf('迭代 %d 数据速率: %.4f\n', jj, C_new);  % 添加此行来显示每次迭代的数据速率
    end
    
    % 计算并保存这一层的平均数据速率
    R_layer(cc) = mean(R_MonteCarlo);
    execution_time = toc;
    fprintf('第 %d 层完成, 用时: %.2f 秒\n', cc, execution_time);
    fprintf('第 %d 层平均数据速率: %.4f\n', cc, R_layer(cc));

end

% 修改K值测试部分
K_values = 2:8;  % 测试2到8个用户
R_K = zeros(length(K_values), 1);
R_K_all = {};  % 添加R_K_all
fprintf('\n开始不同用户数量测试...\n');

for k_idx = 1:length(K_values)
    K_current = K_values(k_idx);
    fprintf('\n测试用户数量 K=%d:\n', K_current);
    
    % 重置相关变量
    R_MonteCarlo = zeros(MonteCarlo,1);
    gama = zeros(K_current, 1);
    R = zeros(K_current, 1);
    location = [10*(1:K_current).' 10*(1:K_current).'];
    W_T_1 = zeros(N, K_current);
    gradient_temp = zeros(K_current, K_current);
    del = zeros(K_current, 1);
    
    % 添加每个K值的完整优化过程
    for cc = 1:Layer_Max
        fprintf('\n处理第 %d 层:\n', cc);
        L = cc;
        phase_transmit=zeros(N,L);
        d_layer = 5*lambda/L;
        for mm1 = 1 : N
            m_z = ceil(mm1/n_max);
            m_x = mod(mm1-1,n_max)+1;
            for mm2 = 1 : N
                n_z = ceil(mm2/n_max);
                n_x = mod(mm2-1,n_max)+1;
                d_temp  = sqrt( (m_x-n_x)^2 +  (m_z-n_z) ^2 )*d_element_SIM;
                d_temp2 = sqrt(d_layer^2 + d_temp^2);
                W_T(mm2,mm1) = lambda^2/4*(d_layer/d_temp2/d_temp2*(1/2/pi/d_temp2-1i/lambda))*exp(1i*2*pi*d_temp2/lambda);
                Corr_T(mm2,mm1) = sinc(2*d_temp/lambda);
            end
        end
        for mm = 1:N
            m_y = ceil(mm/n_max);
            m_x = mod(mm-1,n_max)+1;
            for nn = 1:K_current
                d_transmit = sqrt(d_layer^2 + ...
                    ( (m_x-(1+n_max)/2)*d_element_SIM - (nn-(1+K_current)/2)*lambda/2)^2 + ...
                    ( (m_y-(1+n_max)/2)*d_element_SIM )^2 );
                W_T_1(mm,nn) = lambda^2/4*(d_layer/d_transmit/d_transmit*(1/2/pi/d_transmit-1i/lambda))*exp(1i*2*pi/d_transmit/lambda);
            end
        end
        
        tic  % 开始计时
        fprintf('正在进行Monte Carlo迭代 (%d 次)...\n', MonteCarlo);
        
        PD_transmit_phase = zeros(N,L);
        d = sqrt((10-5*lambda)^2 + location(:,1).^2+location(:,2).^2);
        pathloss = (lambda/(4*pi))^2./d.^(3.5);
        
        % 简化数据存储结构
        all_G = {};
        all_phase_transmit = {};
        all_R = {};
        all_pathloss = {};
        all_location = {};
        
        for jj = 1:MonteCarlo
	        rng(jj)
            fprintf('Monte Carlo迭代 %d/%d\n', jj, MonteCarlo);
            G_in = sqrt(1/2)*(randn(K_current,N)+1i*randn(K_current,N));
            G = diag(sqrt(pathloss))*G_in*(Corr_T)^(1/2);
            phase_transmit = randn(N,L) + 1i*randn(N,L);
            phase_transmit = phase_transmit./abs(phase_transmit);
            W_SIM = diag(phase_transmit(:,1))*W_T_1;
            for l=1:L-1
                W_SIM = diag(phase_transmit(:,l+1))*W_T*W_SIM;
            end
            H_fit = G*W_SIM;
            p = iwf( Pt, sigma2, H_fit, K_current );
            for ii = 1:K_current
                gama(ii) = (abs(H_fit(ii, ii))^2*p(ii))/(abs(H_fit(ii, :)).^2*p - abs(H_fit(ii, ii))^2*p(ii) + sigma2);
                R(ii) = log2(1+gama(ii));
            end
            C_old = sum(R);
            C_new = C_old*2;
            phase_phase_transmit = angle(phase_transmit);
            count = 1;
            while abs(C_new-C_old) >= C_old * 0.000001 && count <= 50
                C_old = C_new;
                for ll = 1:L
                    for mm = 1:N
                        X_left = W_T_1;
                        for ll_left = 1:ll-1
                            X_left = W_T*diag(phase_transmit(:,ll_left))*X_left;
                        end
                        X_right = G;
                        for ll_right = 1:(L-ll)
                            X_right = X_right*diag(phase_transmit(:,L+1-ll_right))*W_T;
                        end
                        for ss1 = 1:K_current
                            del(ss1) = 1/(abs(H_fit(ss1,:)).^2*p+sigma2);
                            for ss2 = 1:K_current
                                temp1 = X_right(ss1,mm)*X_left(mm,ss2);
                                if ss2 == ss1
                                    gradient_temp(ss1,ss2) = del(ss1)*p(ss1)*imag((phase_transmit(mm,ll)*temp1)'*(H_fit(ss1,ss2)));
                                else
                                    gradient_temp(ss1,ss2) = -del(ss1)*gama(ss1)*p(ss2)*imag((phase_transmit(mm,ll)*temp1)'*(H_fit(ss1,ss2)));
                                end
                            end
                        end
                        PD_transmit_phase(mm,ll) = 2/log(2)*sum(sum(gradient_temp));
                    end
                end
                yy = pi/max(max(PD_transmit_phase));
                C_old_1 = C_old;
                C_new = 0;
                count_2 = 1;
                phase_transmit_temp = phase_transmit;
                while C_new < C_old_1 && count_2 <= 20
                    phase_phase_transmit_temp = phase_phase_transmit+yy*PD_transmit_phase;
                    phase_transmit_temp = exp(1i*phase_phase_transmit_temp);
                    W_SIM = diag(phase_transmit_temp(:,1))*W_T_1;
                    for l=1:L-1
                        W_SIM = diag(phase_transmit_temp(:,l+1))*W_T*W_SIM;
                    end
                    H_fit = G*W_SIM;
                    p = iwf( Pt, sigma2, H_fit, K_current );
                    for ii = 1:K_current
                        gama(ii) = (abs(H_fit(ii,ii))^2*p(ii))/(abs(H_fit(ii,:)).^2*p - abs(H_fit(ii,ii))^2*p(ii) + sigma2);
                        R(ii) = log2(1+gama(ii));
                    end
                    C_new = sum(R);
                    yy = yy*0.5;
                    count_2 = count_2+1;
                end
                phase_transmit = phase_transmit_temp;
                phase_phase_transmit = angle(phase_transmit);
                count = count+ 1;
                
                % 使用cell数组保存数据
                all_G{end+1} = G;
                all_phase_transmit{end+1} = phase_transmit;
                all_R{end+1} = R;
                all_pathloss{end+1} = pathloss;
                all_location{end+1} = location;
            end
            R_MonteCarlo(jj) = C_new;
            fprintf('迭代 %d 数据速率: %.4f\n', jj, C_new);  % 添加此行来显示每次迭代的数据速率
        end
        
        % 计算并保存这一层的平均数据速率
        R_layer(cc) = mean(R_MonteCarlo);
        execution_time = toc;
        fprintf('第 %d 层完成, 用时: %.2f 秒\n', cc, execution_time);
        fprintf('第 %d 层平均数据速率: %.4f\n', cc, R_layer(cc));
    end
    
    % 保存当前K值的结果
    R_K_all{k_idx}.K = K_current;
    R_K_all{k_idx}.R = mean(R_MonteCarlo);
    R_K_all{k_idx}.data = R_MonteCarlo;
end

% 显示最终结果
fprintf('\n优化完成!\n');
fprintf('总用时: %.2f 秒\n', toc);
fprintf('最终平均数据速率: %.4f\n', mean(R_layer));  % 添加此行显示总平均数据速率

figure
plot(R_layer)

% 保存所有数据到mat文件
save('sim_ga_data.mat', 'all_G', 'all_phase_transmit', 'all_R', ...
     'all_pathloss', 'all_location', 'W_T', 'W_T_1', ...
     'R_layer', 'R_MonteCarlo', 'R_K', 'R_K_all', ...  % 添加R_K_all
     'Pt', 'sigma2', 'K', 'N', 'execution_time');

% 保存最终数据时添加更多信息
final_data.total_time = toc;
final_data.timestamp = datestr(now);