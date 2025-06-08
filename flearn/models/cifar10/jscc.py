# -*- coding: utf-8 -*-
"""
联合信源信道编码(Joint Source-Channel Coding, JSCC)模型
用于CIFAR10数据集的深度学习实现

JSCC是一种将信源编码和信道编码统一设计的通信技术，
通过端到端的深度学习方法，实现图像压缩和信道传输的联合优化。

Created on Tue Dec  11:00:00 2023
@author: chun
"""

import torch
import torch.nn as nn
from tqdm import trange
from flearn.models.cifar10.channel import Channel
from torch.utils.data import DataLoader, TensorDataset
# from channel import Channel

def image_normalization(norm_type):
    def _inner(tensor: torch.Tensor):
        if norm_type == 'normalization':
            return tensor / 255.0
        elif norm_type == 'denormalization':
            return tensor * 255.0
        else:
            raise Exception('Unknown type of normalization')
    return _inner

def ratio2filtersize(x: torch.Tensor, ratio):
    """
    根据压缩比计算编码器输出的通道数
    
    Args:
        x (torch.Tensor): 输入张量，形状为[batch, channel, height, width]或[channel, height, width]
        ratio (float): 压缩比，表示编码后数据量与原始数据量的比值
    
    Returns:
        int: 编码器最后一层的输出通道数c
    
    工作原理：
        1. 计算输入数据的总像素数
        2. 使用临时编码器计算编码后的空间维度
        3. 根据压缩比计算所需的通道数：c = (原始大小 * 压缩比) / 编码后空间大小
    """
    if x.dim() == 4:
        # 4维张量：[batch_size, channels, height, width]
        # 计算除batch_size外的所有维度大小
        before_size = torch.prod(torch.tensor(x.size()[1:]))
    elif x.dim() == 3:
        # 3维张量：[channels, height, width]
        # 计算所有维度大小
        before_size = torch.prod(torch.tensor(x.size()))
    else:
        raise Exception('Unknown size of input')
    
    # 创建临时编码器计算编码后的空间维度
    encoder_temp = _Encoder(is_temp=True)
    z_temp = encoder_temp(x)
    
    # 根据压缩比计算通道数
    # c = (原始数据大小 * 压缩比) / 编码后的空间大小
    c = before_size * ratio / torch.prod(torch.tensor(z_temp.size()[-2:]))
    return int(c)


class _ConvWithPReLU(nn.Module):
    """
    带有参数化ReLU激活函数的卷积层
    
    PReLU相比普通ReLU的优势：
    - 允许负值有小的正斜率，避免"死神经元"问题
    - 参数可学习，提供更好的表达能力
    """
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        """
        初始化卷积+PReLU层
        
        Args:
            in_channels (int): 输入通道数
            out_channels (int): 输出通道数
            kernel_size (int): 卷积核大小
            stride (int): 步长，默认1
            padding (int): 填充大小，默认0
        """
        super(_ConvWithPReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.prelu = nn.PReLU()

        # 使用Kaiming初始化，适用于PReLU/LeakyReLU激活函数
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='leaky_relu')

    def forward(self, x):
        """前向传播：卷积 → PReLU激活"""
        x = self.conv(x)
        x = self.prelu(x)
        return x


class _TransConvWithPReLU(nn.Module):
    """
    带有激活函数的转置卷积层（反卷积层）
    
    用于解码器中进行上采样和特征重建
    """
    
    def __init__(self, in_channels, out_channels, kernel_size, stride, 
                 activate=None, padding=0, output_padding=0):
        """
        初始化转置卷积+激活层
        
        Args:
            in_channels (int): 输入通道数
            out_channels (int): 输出通道数
            kernel_size (int): 卷积核大小
            stride (int): 步长
            activate (nn.Module): 激活函数，默认PReLU
            padding (int): 填充大小，默认0
            output_padding (int): 输出填充，用于精确控制输出尺寸
        """
        super(_TransConvWithPReLU, self).__init__()
        self.transconv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride, padding, output_padding)
        
        if activate is None:
            self.activate = nn.PReLU()  # 默认使用PReLU激活
        else:
            self.activate = activate
        
        # 根据激活函数类型选择不同的权重初始化方法
        if isinstance(activate, nn.PReLU):
            # PReLU使用Kaiming初始化
            nn.init.kaiming_normal_(self.transconv.weight, mode='fan_out',
                                    nonlinearity='leaky_relu')
        else:
            # 其他激活函数使用Xavier初始化
            nn.init.xavier_normal_(self.transconv.weight)

    def forward(self, x):
        """前向传播：转置卷积 → 激活函数"""
        x = self.transconv(x)
        x = self.activate(x)
        return x


class _Encoder(nn.Module):
    """
    JSCC编码器网络
    
    功能：
    1. 将输入图像编码为低维表示
    2. 进行功率归一化以满足信道传输要求
    3. 实现图像压缩和信道编码的联合优化
    """
    
    def __init__(self, c=1, is_temp=False, P=1):
        """
        初始化编码器
        
        Args:
            c (int): 编码器输出通道数，影响压缩比
            is_temp (bool): 是否为临时编码器（用于计算维度）
            P (float): 发射功率约束，默认1
        """
        super(_Encoder, self).__init__()
        self.is_temp = is_temp
        
        # 编码器网络结构：5层卷积
        # 第1-2层：下采样，减少空间维度
        self.conv1 = _ConvWithPReLU(in_channels=3, out_channels=16, 
                                    kernel_size=5, stride=2, padding=2)
        self.conv2 = _ConvWithPReLU(in_channels=16, out_channels=32, 
                                    kernel_size=5, stride=2, padding=2)
        
        # 第3-4层：特征提取，保持空间维度
        self.conv3 = _ConvWithPReLU(in_channels=32, out_channels=32,
                                    kernel_size=5, padding=2)
        self.conv4 = _ConvWithPReLU(in_channels=32, out_channels=32, 
                                    kernel_size=5, padding=2)
        
        # 第5层：输出层，通道数为2*c（实部和虚部）
        self.conv5 = _ConvWithPReLU(in_channels=32, out_channels=2*c, 
                                    kernel_size=5, padding=2)
        
        # 功率归一化层
        self.norm = self._normlizationLayer(P=P)
        
        # 显式设置所有卷积层参数的requires_grad=True
        for name, param in self.named_parameters():
            if 'conv' in name or 'prelu' in name:
                param.requires_grad = True

    @staticmethod
    def _normlizationLayer(P=1):
        """
        创建功率归一化层
        
        Args:
            P (float): 发射功率约束
        
        Returns:
            function: 归一化函数
            
        功能：
            将编码后的信号进行功率归一化，确保满足信道的功率约束
            归一化公式：z_norm = sqrt(P*k) * z / sqrt(z^T * z)
            其中k是信号维度，P是功率约束
        """
        def _inner(z_hat: torch.Tensor):
            if z_hat.dim() == 4:
                # 批量处理
                batch_size = z_hat.size()[0]
                k = torch.prod(torch.tensor(z_hat.size()[1:]))
            elif z_hat.dim() == 3:
                # 单个样本
                batch_size = 1
                k = torch.prod(torch.tensor(z_hat.size()))
            else:
                raise Exception('Unknown size of input')
            
            # 计算功率归一化
            z_temp = z_hat.reshape(batch_size, 1, 1, -1)    # 转换为行向量
            z_trans = z_hat.reshape(batch_size, 1, -1, 1)   # 转换为列向量
            # 计算归一化后的张量：sqrt(P*k) * z / ||z||
            tensor = torch.sqrt(P * k) * z_hat / torch.sqrt((z_temp @ z_trans))
            
            if batch_size == 1:
                return tensor.squeeze(0)
            return tensor
        return _inner

    def forward(self, x):
        """
        编码器前向传播
        
        Args:
            x (torch.Tensor): 输入图像，形状[batch, 3, H, W]
        
        Returns:
            torch.Tensor: 编码后的特征表示
        """
        # 逐层卷积编码
        x = self.conv1(x)    # [batch, 16, H/2, W/2]
        x = self.conv2(x)    # [batch, 32, H/4, W/4]
        x = self.conv3(x)    # [batch, 32, H/4, W/4]
        x = self.conv4(x)    # [batch, 32, H/4, W/4]
        
        # 如果不是临时编码器，进行最终编码和归一化
        if not self.is_temp:
            x = self.conv5(x)    # [batch, 2*c, H/4, W/4]
            x = self.norm(x)     # 功率归一化
        return x


class _Decoder(nn.Module):
    """
    JSCC解码器网络
    
    功能：
    1. 将编码后的低维表示重建为原始图像
    2. 通过转置卷积进行上采样
    3. 使用Sigmoid激活确保输出在[0,1]范围内
    """
    
    def __init__(self, c=1):
        """
        初始化解码器
        
        Args:
            c (int): 编码器输出通道数，需与编码器保持一致
        """
        super(_Decoder, self).__init__()
        
        # 解码器网络结构：5层转置卷积
        # 第1-3层：特征重建，保持空间维度
        self.tconv1 = _TransConvWithPReLU(
            in_channels=2*c, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.tconv2 = _TransConvWithPReLU(
            in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.tconv3 = _TransConvWithPReLU(
            in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2)
        
        # 第4-5层：上采样，恢复原始图像尺寸
        self.tconv4 = _TransConvWithPReLU(
            in_channels=32, out_channels=16, kernel_size=5, stride=2, 
            padding=2, output_padding=1)
        self.tconv5 = _TransConvWithPReLU(
            in_channels=16, out_channels=3, kernel_size=5, stride=2, 
            padding=2, output_padding=1, activate=nn.Sigmoid())
        
        # 显式设置所有转置卷积层参数的requires_grad=True
        for name, param in self.named_parameters():
            if 'tconv' in name or 'prelu' in name:
                param.requires_grad = True

    def forward(self, x):
        """
        解码器前向传播
        
        Args:
            x (torch.Tensor): 编码后的特征，形状[batch, 2*c, H/4, W/4]
        
        Returns:
            torch.Tensor: 重建的图像，形状[batch, 3, H, W]
        """
        x = self.tconv1(x)    # [batch, 32, H/4, W/4]
        x = self.tconv2(x)    # [batch, 32, H/4, W/4]
        x = self.tconv3(x)    # [batch, 32, H/4, W/4]
        x = self.tconv4(x)    # [batch, 16, H/2, W/2]
        x = self.tconv5(x)    # [batch, 3, H, W] + Sigmoid激活
        return x


class Model(nn.Module):
    """
    完整的JSCC模型
    
    组件：
    1. 编码器：图像压缩和信道编码
    2. 信道模拟：添加噪声模拟真实传输环境
    3. 解码器：信道解码和图像重建
    
    这是一个端到端的深度学习模型，用于联合优化图像压缩和信道传输
    """
    
    def __init__(self, c, channel_type='AWGN', snr=None):
        """
        初始化JSCC模型
        
        Args:
            c (int): 编码器输出通道数，决定压缩比
            channel_type (str): 信道类型，默认'AWGN'（加性白高斯噪声）
            snr (float): 信噪比，如果为None则不添加信道噪声
        """
        super(Model, self).__init__()
        # 创建模型
        self.encoder = _Encoder(c=c)
        
        # 如果指定了SNR，创建信道模型
        if snr is not None:
            self.channel = Channel(channel_type, snr)
        
        self.decoder = _Decoder(c=c)
        
        # 添加联邦学习所需的属性
        self.size = self._get_model_size()  # 模型参数大小（字节）
        self.flops = self._estimate_flops()
        

    def forward(self, x):
        """
        JSCC模型前向传播
        
        Args:
            x (torch.Tensor): 输入图像，形状[batch, 3, H, W]
        
        Returns:
            torch.Tensor: 重建的图像，形状[batch, 3, H, W]
        
        流程：
            输入图像 → 编码器 → 信道(可选) → 解码器 → 重建图像
        """
        # 编码阶段
        z = self.encoder(x)
        
        # 信道传输阶段（如果存在信道模型）
        if hasattr(self, 'channel') and self.channel is not None:
            z = self.channel(z)
        
        # 解码阶段
        x_hat = self.decoder(z)
        return x_hat

    def change_channel(self, channel_type='AWGN', snr=None):
        """
        动态改变信道参数
        
        Args:
            channel_type (str): 新的信道类型
            snr (float): 新的信噪比，None表示无噪声
        """
        if snr is None:
            self.channel = None
        else:
            self.channel = Channel(channel_type, snr)

    def get_channel(self):
        """
        获取当前信道配置
        
        Returns:
            dict or None: 信道配置信息
        """
        if hasattr(self, 'channel') and self.channel is not None:
            return self.channel.get_channel()
        return None

    def loss(self, prd, gt):
        """
        计算重建损失
        
        Args:
            prd (torch.Tensor): 预测的重建图像
            gt (torch.Tensor): 真实图像
        
        Returns:
            torch.Tensor: MSE损失值
        """
        criterion = nn.MSELoss(reduction='mean')
        loss = criterion(prd, gt)
        return loss

    def _get_model_size(self):
        """计算模型参数大小（字节）"""
        return sum(p.numel() * 4 for p in self.parameters())  # 假设float32

    def _estimate_flops(self):
        """估算模型FLOPs"""
        return 1e8  # 简化估算，实际可用更精确的方法
    
    
    # def train_epoch(self, model, optimizer, param, data_loader):
    #     self.train()
    #     epoch_loss = 0

    #     for iter, (images, _) in enumerate(data_loader):
    #         images = images.cuda() if param['parallel'] and torch.cuda.device_count(
    #         ) > 1 else images.to(param['device'])
    #         optimizer.zero_grad()
    #         outputs = model.forward(images)
    #         outputs = image_normalization('denormalization')(outputs)
    #         images = image_normalization('denormalization')(images)
    #         loss = model.loss(images, outputs) if not param['parallel'] else model.module.loss(
    #             images, outputs)
    #         loss.backward()
    #         optimizer.step()
    #         epoch_loss += loss.detach().item()
    #     epoch_loss /= (iter + 1)

    #     return epoch_loss, optimizer

    # ============ 联邦学习接口方法 ============
    def set_params(self, model_params):
        """设置模型参数（联邦学习接口）"""
        if isinstance(model_params, dict):
            self.load_state_dict(model_params)
        else:
            # 如果是参数列表，需要转换为state_dict格式
            state_dict = {}
            param_iter = iter(model_params)
            for name, param in self.named_parameters():
                state_dict[name] = next(param_iter)
            self.load_state_dict(state_dict)

    def get_params(self):
        """获取模型参数（联邦学习接口）"""
        return self.parameters()
    
    def solve_inner(self, data, optimizer, num_epochs=1, batch_size=32):
        '''执行多轮训练'''
        self.train()
        
        x_tensor = torch.tensor(data['x'], dtype=torch.float32)
        y_tensor = torch.tensor(data['y'], dtype=torch.float32)
        # 创建数据集
        dataset = TensorDataset(x_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        for _ in trange(num_epochs, desc='Epoch', leave=False, ncols=120):
            for X, y in dataloader:
                optimizer.zero_grad()
                self.zero_grad()  # 清除梯度
                outputs = self.forward(X)
                outputs = image_normalization('denormalization')(outputs)
                y = image_normalization('denormalization')(y)
                loss = self.loss(outputs, y)
                loss.backward()
                optimizer.step()
                
        soln = self.get_params()
        return soln
    
    def solve_iters(self, data, optimizer, num_iters=1, batch_size=32):
        '''执行指定迭代次数的训练'''
        self.train()
    
        x_tensor = torch.tensor(data['x'], dtype=torch.float32)
        y_tensor = torch.tensor(data['y'], dtype=torch.float32)
        # 创建数据集
        dataset = TensorDataset(x_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        iter_count = 0
        for X, y in dataloader:
            if iter_count >= num_iters:
                break
            optimizer.zero_grad()
            outputs = self.model(X)
            outputs = image_normalization('denormalization')(outputs)
            y = image_normalization('denormalization')(y)
            loss = self.criterion(outputs, y)
            loss.backward()
            optimizer.step()
            iter_count += 1
        
        soln = self.get_params()

        return soln
    
    def get_gradients(self):
        """获取模型梯度（联邦学习接口）"""
        grads = []
        for p in self.parameters():
            if p.grad is not None:
                grads.append(p.grad.clone().detach().flatten())
            else:
                grads.append(torch.zeros_like(p.data).flatten())
        return grads
    
        

    def cal_gradients(self, data, optimizer):
        """计算梯度（联邦学习接口）"""
        self.train()
        optimizer.zero_grad()
        
        X, y = data['x'], data['y']
        
        # 转换数据类型
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, dtype=torch.float32)
        
        # 前向传播和损失计算
        self.zero_grad()
        output = self.forward(X)
        loss = self.loss(output, y)
        
        # 反向传播
        loss.backward()
        
        # 收集梯度
        gradients = []
        for name, param in self.named_parameters():
            if isinstance(param, torch.Tensor) and param.grad is not None:
                gradients.append(param.grad.clone().flatten())

        return gradients
    
    def test(self, data):
        '''评估模型性能'''
        self.eval()
        
        with torch.no_grad():
            X = data['x']
            y = data['y']

            # 前向传播
            outputs = self.forward(X)
            
            # 反归一化
            outputs_denorm = image_normalization('denormalization')(outputs)
            images_denorm = image_normalization('denormalization')(y)
            
            # 计算损失
            loss = self.loss(outputs_denorm, images_denorm)
            epoch_loss = loss.item()
            
            # 计算PSNR
            mse = torch.mean((outputs_denorm - images_denorm) ** 2)
            if mse == 0:
                epoch_psnr = float('inf')
            else:
                epoch_psnr = 20 * torch.log10(255.0 / torch.sqrt(mse))
                epoch_psnr = epoch_psnr.item()

        return epoch_psnr
    
    def train_error_and_loss(self, data):
        """
        计算训练集上的PSNR和损失
        
        Args:
            data (dict): 包含 'x' 和 'y' 键的数据字典
        
        Returns:
            tuple: (PSNR值, 损失值)
        """
        self.train()
        
        with torch.no_grad():
            X = data['x']
            y = data['y']

            # 前向传播
            outputs = self.forward(X)
            
            # 反归一化
            outputs_denorm = image_normalization('denormalization')(outputs)
            images_denorm = image_normalization('denormalization')(y)
            
            # 计算损失
            loss = self.loss(outputs_denorm, images_denorm)
            epoch_loss = loss.item()
            
            # 计算PSNR
            mse = torch.mean((outputs_denorm - images_denorm) ** 2)
            if mse == 0:
                epoch_psnr = float('inf')
            else:
                epoch_psnr = 20 * torch.log10(255.0 / torch.sqrt(mse))
                epoch_psnr = epoch_psnr.item()

        return epoch_psnr, epoch_loss
    


# ============ 测试代码 ============
if __name__ == '__main__':
    # 创建模型实例
    model = Model(c=20)
    print("模型结构：")
    
    for name, param in model.named_parameters():
        print(name, param.size())
    
    print(f"\n模型参数数量: ", len(list(model.named_parameters())))
    
    print(f"模型参数总数: ", len(model.state_dict()))
    
    # 测试前向传播
    x = torch.rand(1, 3, 128, 128)  # 模拟CIFAR10输入（放大版）
    y = model(x)
    print(f"\n输入尺寸: {x.size()}")
    print(f"输出尺寸: {y.size()}")
    print(f"输出数值范围: [{y.min():.4f}, {y.max():.4f}]")
    
    # 测试归一化层
    print(f"\n归一化层函数: {model.encoder.norm}")
    norm_output = model.encoder.norm(y)
    print(f"归一化后尺寸: {norm_output.size()}")