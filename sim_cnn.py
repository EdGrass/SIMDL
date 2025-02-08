# 检查所需的包导入
import os                     # 添加 os 导入
import numpy as np              # 需要 numpy
import torch                    # 需要 pytorch
import torch.nn as nn          # pytorch的一部分
import scipy.io as sio         # 需要 scipy
from torch.utils.data import Dataset, DataLoader  # pytorch的一部分
import matplotlib.pyplot as plt # 需要 matplotlib
from sklearn.model_selection import train_test_split  # 需要 scikit-learn
from torch.optim import lr_scheduler  # 修改这行

class PhaseDataset(Dataset):
    def __init__(self, data_path, mode='train'):
        try:
            data = sio.loadmat(data_path)
            # 处理数据并检查NaN
            self.G = np.array([item for item in data['all_G'][0]], dtype=np.complex128)
            self.phase_transmit = np.array([item for item in data['all_phase_transmit'][0]], dtype=np.complex128)
            self.pathloss = np.array([item for item in data['all_pathloss'][0]])
            self.location = np.array([item for item in data['all_location'][0]])
            
            # 检查数据是否包含NaN
            if np.any(np.isnan(self.G)) or np.any(np.isnan(self.phase_transmit)):
                raise ValueError("输入数据包含NaN值")
                
        except Exception as e:
            raise Exception(f"加载数据失败: {str(e)}")
            
        # 提取并预处理特征
        features_list = []
        labels_list = []
        
        for i in range(len(self.G)):
            # 安全地将复数转换为实数特征
            G_abs = np.abs(self.G[i])
            G_angle = np.angle(self.G[i])
            
            # 检查并处理可能的无效值
            G_abs = np.nan_to_num(G_abs, nan=0.0)
            G_angle = np.nan_to_num(G_angle, nan=0.0)
            
            features = np.concatenate([
                G_abs.flatten(),
                G_angle.flatten(),
                np.nan_to_num(self.pathloss[i].flatten(), nan=0.0),
                np.nan_to_num(self.location[i].flatten(), nan=0.0)
            ])
            
            # 相位标签
            phase = np.angle(self.phase_transmit[i])
            phase = np.nan_to_num(phase, nan=0.0)
            
            features_list.append(features)
            labels_list.append(phase.flatten())
            
        # 转换为numpy数组并进行额外的检查
        self.features = np.array(features_list, dtype=np.float32)
        self.labels = np.array(labels_list, dtype=np.float32)
        
        # 处理特征标准化中可能的除零问题
        eps = 1e-8
        mean = np.mean(self.features, axis=0)
        std = np.std(self.features, axis=0) + eps
        self.features = (self.features - mean) / std
        
        # 确保没有无穷值
        self.features = np.clip(self.features, -1e6, 1e6)
        self.labels = np.clip(self.labels, -np.pi, np.pi)
        
        # 确保标签相位在[-π, π]范围内
        self.labels = ((self.labels + np.pi) % (2 * np.pi)) - np.pi
        
        # 添加数据增强
        if mode == 'train':
            # 随机添加相位偏移进行数据增强
            phase_shift = np.random.uniform(-np.pi, np.pi, size=len(self.labels))
            self.labels = ((self.labels + phase_shift[:, np.newaxis] + np.pi) % (2 * np.pi)) - np.pi
        
        # 训练集/测试集分割
        total_samples = len(self.features)
        split_idx = int(0.8 * total_samples)
        
        if mode == 'train':
            self.features = self.features[:split_idx]
            self.labels = self.labels[:split_idx]
        else:
            self.features = self.features[split_idx:]
            self.labels = self.labels[split_idx:]

    def __len__(self):
        return len(self.features)
        
    def __getitem__(self, idx):
        return (
            torch.FloatTensor(self.features[idx]),
            torch.FloatTensor(self.labels[idx])
        )

class PhaseNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(PhaseNet, self).__init__()
        # 修改网络结构以适应正确的维度
        self.input_norm = nn.BatchNorm1d(input_dim)
        
        # 编码器部分 - 调整维度
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.5)
        )
        
        # 中间层 - 简化结构
        middle_dim = hidden_dim // 2
        self.middle = nn.Sequential(
            nn.Linear(hidden_dim, middle_dim),
            nn.ReLU(),
            nn.BatchNorm1d(middle_dim),
            nn.Dropout(0.5)
        )
        
        # 解码器部分 - 修改激活函数
        self.decoder = nn.Sequential(
            nn.Linear(middle_dim, output_dim),
            nn.Hardtanh()  # 使用Hardtanh替换Tanh以获得更精确的[-1,1]范围
        )
        
        # 初始化权重
        self._initialize_weights()
        
        # 系统参数
        self.Pt = 10
        self.sigma2 = 1e-10

    def _initialize_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # 确保输入维度正确
        if len(x.shape) == 3:  # 如果是批量数据
            x = x.squeeze(1)
        elif len(x.shape) == 1:  # 如果是单个样本
            x = x.unsqueeze(0)
            
        # 添加批归一化
        x = self.input_norm(x)
        
        # 添加残差连接
        identity = x
        x = self.encoder(x)
        if x.shape == identity.shape:  # 如果维度匹配则添加残差
            x = x + identity
            
        x = self.middle(x)
        x = self.decoder(x)
        
        # 确保输出在[-1,1]范围内
        return torch.clamp(x, -1, 1)
    
    def get_rate(self, phase_config, L, K):
        """计算给定相位配置的系统总速率"""
        try:
            # 确保输入是正确维度的批次数据
            if len(phase_config.shape) > 1:
                batch_size = phase_config.shape[0]
                phase_config = phase_config[0]  # 只取第一个样本
            
            # 计算需要的相位数量
            total_phases = L * K * K
            
            # 处理相位维度
            if len(phase_config) > total_phases:
                phase_config = phase_config[:total_phases]
            elif len(phase_config) < total_phases:
                # 填充相位
                padding = torch.zeros(total_phases - len(phase_config))
                phase_config = torch.cat([phase_config, padding])
            
            # 将相位从[-1,1]映射到[-π,π]
            phase = phase_config * np.pi
            
            # 重塑相位为正确的维度并保持梯度信息
            phase = phase.reshape(K, K, L)
            if not phase.requires_grad:
                phase = phase.clone().detach().requires_grad_(True)
            
            # 计算信道矩阵和速率
            H = self._compute_channel_matrix(phase, L, K)
            p = self._compute_power_allocation(H, K)
            rates = self._compute_user_rates(H, p, K)
            
            return torch.sum(rates)
        except Exception as e:
            print(f"速率计算错误: {str(e)}")
            print(f"相位形状: {phase_config.shape}, 需要的相位数: {total_phases}")
            return torch.tensor(0.0, requires_grad=True)
    
    def _compute_channel_matrix(self, phase, L, K):
        """计算系统信道矩阵"""
        try:
            H = torch.zeros((K, K), dtype=torch.complex64)
            
            # 相位已经是(K, K, L)形状
            for i in range(K):
                for j in range(K):
                    # 使用复数指数计算信道增益
                    H[i,j] = torch.sum(torch.exp(1j * phase[i,j,:]))
            return H
        except Exception as e:
            print(f"信道矩阵计算错误: {str(e)}")
            return torch.ones((K, K), dtype=torch.complex64)

    def _compute_power_allocation(self, H, K):
        """计算功率分配"""
        # 使用简单的均匀功率分配
        return torch.ones(K) * (self.Pt / K)
    
    def _compute_user_rates(self, H, p, K):
        """计算每个用户的速率"""
        rates = []
        for k in range(K):
            signal = torch.abs(H[k,k])**2 * p[k]
            interference = torch.sum(torch.abs(H[k,:])**2 * p) - signal
            sinr = signal / (interference + self.sigma2)
            rates.append(torch.log2(1 + sinr))
        return torch.stack(rates)

def train_model(model, train_loader, val_loader, epochs=300, batch_size=128, 
               learning_rate=0.0001, min_lr=1e-6, weight_decay=1e-4):
    """改进的训练函数"""
    # 使用更保守的学习率
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    # 使用简单的学习率调度器
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=20,
        verbose=True
    )
    
    def rate_loss(pred_phase, K=4, L=7):
        """改进的损失函数"""
        try:
            rates = []
            for phase in pred_phase:
                rate = model.get_rate(phase, L, K)
                rates.append(rate)
            batch_rate = torch.mean(torch.stack(rates))
            
            # 减小正则化影响
            l2_reg = sum(torch.sum(p ** 2) for p in model.parameters())
            reg_weight = 1e-6
            
            return -batch_rate + reg_weight * l2_reg
        except Exception as e:
            print(f"损失计算错误: {str(e)}")
            return torch.tensor(0.0, requires_grad=True)
    
    train_rates_history = []
    val_rates_history = []
    best_model_path = 'best_model.pth'
    best_rate = float('-inf')
    
    print(f"\n开始训练，总共 {epochs} 轮...")
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_rates = []
        
        for batch_x, _ in train_loader:
            # 减少数据增强的频率和强度
            if np.random.rand() > 0.7:  # 降低增强频率
                noise = torch.randn_like(batch_x) * 0.02  # 降低噪声强度
                batch_x = batch_x + noise
            
            optimizer.zero_grad()
            pred_phase = model(batch_x)
            loss = rate_loss(pred_phase)
            loss.backward()
            
            # 调整梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_rates.append(-loss.item())
        
        # 验证阶段
        model.eval()
        val_rates = []
        with torch.no_grad():
            for batch_x, _ in val_loader:
                pred_phase = model(batch_x)
                val_rate = -rate_loss(pred_phase)
                val_rates.append(val_rate.item())
        
        # 不使用移动平均，直接计算当前epoch的平均值
        avg_train_rate = np.mean(train_rates)
        avg_val_rate = np.mean(val_rates)
        
        train_rates_history.append(avg_train_rate)
        val_rates_history.append(avg_val_rate)
        
        # 保存最佳模型
        if avg_val_rate > best_rate:
            best_rate = avg_val_rate
            torch.save(model.state_dict(), best_model_path)
        
        # 更新学习率
        scheduler.step(avg_val_rate)
        
        # 打印进度
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}: "
                  f"Train Rate = {avg_train_rate:.2f}, "
                  f"Val Rate = {avg_val_rate:.2f}, "
                  f"LR = {scheduler.optimizer.param_groups[0]['lr']:.6f}")
    
    print("\n训练完成!")
    print(f"最佳验证速率: {best_rate:.2f}")
    
    return train_rates_history, val_rates_history

def plot_learning_curves(train_losses, val_losses, save_path):
    """绘制学习曲线"""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    if val_losses:
        # 调整验证损失的x轴以匹配实际的验证频率
        val_epochs = np.arange(0, len(train_losses), 5)[:len(val_losses)]
        plt.plot(val_epochs, val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Learning Curves')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def evaluate_model(model, test_loader):
    model.eval()
    criterion = nn.MSELoss()
    test_loss = 0
    predictions = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            outputs = model(batch_x)
            test_loss += criterion(outputs, batch_y).item()
            predictions.append(outputs.numpy())
    
    return test_loss/len(test_loader), np.concatenate(predictions)

def check_environment():
    """检查运行环境"""
    try:
        # 检查CUDA可用性
        if torch.cuda.is_available():
            print(f"使用GPU: {torch.cuda.get_device_name(0)}")
            device = "cuda"
        else:
            print("使用CPU")
            device = "cpu"
            
        # 检查数据文件
        if not os.path.exists('sim_ga_data.mat'):
            raise FileNotFoundError("找不到GA算法生成的数据文件")
            
        return device
    except Exception as e:
        raise RuntimeError(f"环境检查失败: {str(e)}")

if __name__ == "__main__":
    try:
        # 修改训练参数
        BATCH_SIZE = 512      # 减小批量大小
        EPOCHS = 400        # 减少轮数防止过拟合
        HIDDEN_DIM = 512     # 减小网络容量
        
        print("运行正常训练模式...")
        print(f"批量大小: {BATCH_SIZE}")
        print(f"训练轮数: {EPOCHS}")
        print(f"隐藏层维度: {HIDDEN_DIM}")
        
        # 设置随机种子
        torch.manual_seed(42)
        np.random.seed(42)
        
        # 检查环境
        device = check_environment()
        
        # 加载数据
        train_dataset = PhaseDataset('sim_ga_data.mat', mode='train')
        val_dataset = PhaseDataset('sim_ga_data.mat', mode='val')
        
        # 创建数据加载器
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
        
        # 初始化模型
        input_dim = train_dataset.features.shape[1]
        output_dim = train_dataset.labels.shape[1]
        model = PhaseNet(input_dim, HIDDEN_DIM, output_dim)
        
        # 在训练时添加进度打印
        print(f"数据集大小: {len(train_dataset)}")
        print(f"批量大小: {BATCH_SIZE}")
        print(f"训练轮数: {EPOCHS}")
        
        # 训练模型并获取训练历史
        train_losses, val_losses = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            learning_rate=0.0001,  # 降低学习率
            min_lr=1e-6,
            weight_decay=1e-4      # 增加权重衰减
        )
        
        # 保存完整的训练历史
        np.save('train_history.npy', {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'training_params': {
                'epochs': EPOCHS,
                'batch_size': BATCH_SIZE,
                'hidden_dim': HIDDEN_DIM
            }
        })
        
        print("模型训练完成!")
        
    except Exception as e:
        print(f"训练过程出错: {str(e)}")
        raise  # 添加raise以显示完整的错误堆栈
