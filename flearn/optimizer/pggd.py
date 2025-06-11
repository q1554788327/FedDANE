import torch
from torch.optim import Optimizer

class PerturbedGradientDescent(Optimizer):
    """
    实现Perturbed Gradient Descent (PGD)优化器，也称为FedDane优化器
    用于联邦学习中的客户端局部训练，结合了近端项和梯度校正机制
    
    参数:
        params (iterable): 需要优化的参数迭代器
        lr (float): 学习率
        mu (float): 近端项系数，控制向全局模型收敛的强度
    """
    def __init__(self, params, lr=0.001, mu=0.01):
        # 检查参数合法性
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if mu < 0.0:
            raise ValueError(f"Invalid mu value: {mu}")
        
        # 初始化超参数
        defaults = dict(lr=lr, mu=mu)
        super(PerturbedGradientDescent, self).__init__(params, defaults)
        
        # 为每个参数创建缓存槽
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                # 存储全局模型参数
                state['vstar'] = torch.zeros_like(p.data)
                # 存储梯度校正项，平均梯度与本地旧梯度的差值
                state['gold'] = torch.zeros_like(p.grad.data) if p.grad is not None else torch.zeros_like(p.data)

    @torch.no_grad()
    def step(self, closure=None):
        """执行单步优化"""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # 遍历参数组和参数
        for group in self.param_groups:
            lr = group['lr']
            mu = group['mu']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                state = self.state[p]
                vstar = state['vstar'] # 存储全局模型参数
                gold = state['gold'] # 
                
                # 计算参数更新
                # w_{t+1} = w_t - lr * (∇F(w_t) + gold + mu * (w_t - vstar))
                # p.data.add_(-lr, p.grad.data + gold + mu * (p.data - vstar))
                p.add_(-lr, p.grad + gold + mu * (p - vstar))
                
        return loss
    
    def set_gradients(self, avg_gradient):
        '''设置优化器梯度为avg_gradient'''
        idx = 0
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    p.grad = torch.zeros_like(p.data)
                # avg_gradient[idx] 可能是扁平化的，需要reshape
                if p.grad.numel() == avg_gradient[idx].numel():
                    p.grad.copy_(avg_gradient[idx].view_as(p.grad))
                else:
                    raise ValueError("Gradient shape mismatch in set_gradients")
                idx += 1
    
    def set_global_params(self, cog, avg_gradient, client=None):
        """设置全局参数和梯度差"""
        # 支持cog为state_dict（dict）或参数列表
        # 先将cog转为参数值列表
        if isinstance(cog, dict):
            cog_values = list(cog.values())
        else:
            cog_values = cog

        # 将全局模型参数按顺序赋值给每个客户端优化器
        idx = 0
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['vstar'].copy_(cog_values[idx])
                idx += 1

        # 获取旧梯度（假设client有获取梯度的方法）
        if client is not None:
            gprev = client.get_grads()
        else:
            gprev = [[torch.zeros_like(p) for p in group['params']] for group in self.param_groups]

        # 计算梯度差 gdiff = avg_gradient - gprev
        gdiff = []
        for group_grad, group_gprev in zip(avg_gradient, gprev):
            group_diff = []
            for g1, g2 in zip(group_grad, group_gprev):
                group_diff.append(g1 - g2)
            gdiff.append(group_diff)

        # 更新gold为梯度差
        for i, group in enumerate(self.param_groups):
            for j, p in enumerate(group['params']):
                state = self.state[p]
                state['gold'].copy_(gdiff[i][j]) 

    def set_gradient_correction(self, avg_gradient, local_gradient):
        """
        设置梯度校正项gold
        gold = ∇F(w_t) - ∇F_old(w_t)
        其中∇F_old是旧模型的梯度
        """
        # 确保梯度列表长度匹配
        if len(avg_gradient) != len(self.param_groups[0]['params']):
            raise ValueError("Gradient list length does not match optimizer parameters")
            
        # 计算并设置梯度校正项
        for i, p in enumerate(self.param_groups[0]['params']):
            state = self.state[p]
            # 计算梯度差
            state['gold'].copy_(avg_gradient[i].data - local_gradient[i].data)