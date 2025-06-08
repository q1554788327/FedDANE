import torch
import torch.nn as nn
import math

def calculate_path_loss(h_uav: float, d_horizontal: float, eta_LoS: float, f_carrier: float) -> float:
    """
    计算空地通信的路径损耗（单位：dB）
    
    参数:
        h_uav (float): 无人机飞行高度（米）
        d_horizontal (float): 无人机与基站的水平距离（米）
        eta_LoS (float): 视距链路的环境损耗（dB）
        f_carrier (float): 载波频率（Hz）
    
    返回:
        float: 路径损耗值（dB）
    """
    # 计算几何距离 sqrt(h^2 + d^2)
    geometric_distance = math.sqrt(h_uav**2 + d_horizontal**2)
    
    # 计算常数项 C = 20*log10(4πf_c / c)
    c = 3e8  # 光速（m/s）
    C = 20 * math.log10((4 * math.pi * f_carrier) / c)
    
    # 路径损耗公式
    path_loss = 20 * math.log10(geometric_distance) + eta_LoS + C
    
    return path_loss


class Channel(nn.Module):
    def __init__(self, channel_type='AWGN', snr=20, h_uav=100.0, d_horizontal=500.0, eta_LoS=1.0, f_carrier=2.4e9):
        if channel_type not in ['AWGN', 'Rayleigh', 'CustomChannel']:
            raise Exception('Unknown type of channel')
        super(Channel, self).__init__()
        self.channel_type = channel_type
        self.snr = snr
        
        self.h_uav = h_uav
        self.d_horizontal = d_horizontal
        self.eta_LoS = eta_LoS
        self.f_carrier = f_carrier
        

    def forward(self, z_hat):
        if z_hat.dim() not in {3, 4}:
            raise ValueError('Input tensor must be 3D or 4D')
        
        # if z_hat.dim() == 4:
        #     # k = np.prod(z_hat.size()[1:])
        #     k = torch.prod(torch.tensor(z_hat.size()[1:]))
        #     sig_pwr = torch.sum(torch.abs(z_hat).square(), dim=(1, 2, 3), keepdim=True) / k
        # elif z_hat.dim() == 3:
        #     # k = np.prod(z_hat.size())
        #     k = torch.prod(torch.tensor(z_hat.size()))
        #     sig_pwr = torch.sum(torch.abs(z_hat).square()) / k
            
        if z_hat.dim() == 3:
            z_hat = z_hat.unsqueeze(0)
        
        k = z_hat[0].numel()
        sig_pwr = torch.sum(torch.abs(z_hat).square(), dim=(1, 2, 3), keepdim=True) / k    
        noi_pwr = sig_pwr / (10 ** (self.snr / 10))
        noise = torch.randn_like(z_hat) * torch.sqrt(noi_pwr/2)
        
        # 克隆输入以避免原地操作
        z_hat = z_hat.clone()
        
        if self.channel_type == 'Rayleigh':
            # hc = torch.randn_like(z_hat)  wrong implement before
            # hc = torch.randn(1, device = z_hat.device) 
            hc = torch.randn(2, device = z_hat.device) 
        
            # clone for in-place operation  
            z_hat = z_hat.clone()
            z_hat[:,:z_hat.size(1)//2] = hc[0] * z_hat[:,:z_hat.size(1)//2]
            z_hat[:,z_hat.size(1)//2:] = hc[1] * z_hat[:,z_hat.size(1)//2:]
        
        # 论文中定义的信道模型
        elif self.channel_type == 'CustomChannel':
            # 获取路径损耗计算所需参数
            # h_uav = self.channel_params.get('h_uav', 100.0)  # 默认100米高度
            # d_horizontal = self.channel_params.get('d_horizontal', 500.0)  # 默认500米水平距离
            # eta_LoS = self.channel_params.get('eta_LoS', 1.0)  # 默认视距损耗1dB
            # f_carrier = self.channel_params.get('f_carrier', 2.4e9)  # 默认2.4GHz
            
            # 计算路径损耗(dB)
            path_loss_db = calculate_path_loss(self.h_uav, self.d_horizontal, self.eta_LoS, self.f_carrier)
            
            # 将dB转换为线性比例
            path_loss_linear = 10 ** (-path_loss_db / 10)
            
            # 应用路径损耗衰减
            z_hat = z_hat * path_loss_linear
            
            

            # z_hat = hc * z_hat

        return z_hat + noise

    def get_channel(self):
        if self.channel_type == 'CustomChannel':
            return self.channel_type, self.snr, self.channel_params
        else:
            return self.channel_type, self.snr


if __name__ == '__main__':
    # test
    channel = Channel(channel_type='AWGN', snr=10)
    z_hat = torch.randn(64, 10, 5, 5)
    z_hat = channel(z_hat)
    print(z_hat)

    channel = Channel(channel_type='Rayleigh', snr=10)
    z_hat = torch.randn(10, 5, 5)
    z_hat = channel(z_hat)
    print(z_hat)
