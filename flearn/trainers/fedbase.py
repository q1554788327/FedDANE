import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import copy
from flearn.models.cifar10.jscc import Model
from flearn.optimizer.pggd import PerturbedGradientDescent  
from flearn.models.client import Client
from flearn.utils.model_utils import Metrics


class BaseFedarated(object):
    """
    联邦学习基类，提供联邦学习的基本功能
    
    主要功能：
    1. 客户端管理和初始化
    2. 模型参数聚合
    3. 客户端选择策略
    4. 模型测试和评估
    5. 梯度可视化和分析
    """
    
    def __init__(self, params, learner, dataset):
        """
        初始化联邦学习基类
        
        Args:
            params (dict): 参数配置字典
            learner (class): 模型类
            dataset (tuple): 数据集元组 (users, groups, train_datasets, test_datasets)
        """
        # 将参数字典中的所有键值对转换为类属性
        for key, val in params.items(): 
            setattr(self, key, val)
        
        # 设置设备（CPU或GPU）
        self.device = params['device']
        print(f"Using device: {self.device}")
        
        # 初始化全局模型
        print("model_params len:", len(params['model_params']))
        self.global_model = learner(*params['model_params'])
            
        self.global_model.to(self.device)
        
        # 设置随机种子以保证实验可重现
        if hasattr(self, 'seed'):
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)
        
        # 基于数据集创建客户端实例
        self.clients = self.setup_clients(dataset, params, learner, self.global_model)
        print('{} Clients in Total'.format(len(self.clients)))
        
        # 初始化系统性能指标记录器
        self.metrics = Metrics(self.clients, params)

    def __del__(self):
        """析构函数：清理资源"""
        # PyTorch不需要显式关闭模型，但可以清理GPU缓存
        if hasattr(self, 'client_model'):
            del self.global_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def setup_clients(self, dataset, params, learner, model=None):
        """
        根据全局模型初始化所有客户端
        
        Args:
            dataset (tuple): 包含(users, groups, train_data, test_data)的元组
            model: 客户端使用的模型实例
            
        Returns:
            list: 客户端对象列表
        """
        users, groups, train_datasets, test_datasets = dataset
        
        # 如果没有组信息，为每个用户分配None
        if len(groups) == 0:
            groups = [None for _ in users]
        
        # 为每个用户创建客户端实例
        all_clients = []
        for u, g in zip(users, groups):
            client_model = copy.deepcopy(self.global_model)  # 深拷贝全局模型 
            inner_opt = PerturbedGradientDescent(
                params=client_model.parameters(),  # 传入模型的所有可训练参数
                lr=params['learning_rate'],   # 本地学习率
                mu=params['mu']               # 近端项系数，控制向全局模型收敛的强度
            )

            client = Client(u, g, train_datasets[u], test_datasets[u], client_model, inner_opt)
            all_clients.append(client)
        
        return all_clients

    def train_error_and_loss(self):
        """
        计算所有客户端在训练集上的PSNR和损失
        
        Returns:
            tuple: (客户端ID列表, 组列表, PSNR列表, 损失列表)
        """
        psnrs = []         # 每个客户端的PSNR值
        losses = []         # 每个客户端的损失值

        # 遍历所有客户端，计算训练误差和损失
        for c in self.clients:
            # 获取客户端训练集上的性能指标
            cp, cl = c.train_error_and_loss() 
            psnrs.append(cp)        # 样本数
            losses.append(cl * 1.0)       # 损失值
        
        # 收集客户端ID和组信息
        ids = [c.id for c in self.clients]
        groups = [c.group for c in self.clients]

        return ids, groups, psnrs, losses

    def show_grads(self):  
        """
        显示所有客户端的梯度和全局梯度
        
        用于梯度分析和可视化，帮助理解联邦学习中的梯度分布情况
        
        Returns:
            list: 包含所有客户端梯度和全局梯度的列表
        """
        # 计算模型参数总数
        model_len = sum(p.numel() for p in self.global_model.parameters())
        global_grads = torch.zeros(model_len, device=self.device)

        intermediate_grads = []  # 存储中间梯度
        samples = []            # 存储样本数
        
        # 收集每个客户端的梯度
        for c in self.clients:
            # 获取客户端梯度
            client_grads = c.get_grads()
            
            num_samples = c.num_samples
            samples.append(num_samples)
            
            # 将客户端梯度转换为PyTorch张量
            if not isinstance(client_grads, torch.Tensor):
                client_grads = torch.tensor(client_grads, device=self.device)
            
            # 累加加权梯度到全局梯度
            global_grads += client_grads * num_samples
            intermediate_grads.append(client_grads.cpu().numpy())

        # 计算全局平均梯度
        total_samples = sum(samples)
        if total_samples > 0:
            global_grads = global_grads / total_samples
        
        # 添加全局梯度到结果列表
        intermediate_grads.append(global_grads.cpu().numpy())

        return intermediate_grads

    def test(self):
        """
        在给定客户端上测试最新模型
        
        Returns:
            tuple: (客户端ID列表, 组列表,PSNR列表)
        """
        psnrs = []          # 每个客户端的PSNR值
        
        # 设置客户端模型参数为最新的全局模型参数
        # self.global_model.set_params(self.latest_model)
        
        # 遍历所有客户端进行测试
        for c in self.clients:
            # 获取客户端测试结果
            cp = c.test()  # 返回psnr值
            psnrs.append(cp)  # 正确预测数
        
        # 收集客户端ID和组信息
        ids = [c.id for c in self.clients]
        groups = [c.group for c in self.clients]
        
        return ids, groups, psnrs

    def save(self):
        """
        保存模型和训练状态
        
        子类可以重写此方法以实现具体的保存逻辑
        """
        # 这里可以实现模型保存逻辑
        # 例如：torch.save(self.latest_model, 'model.pth')
        pass

    def select_clients(self, round, num_clients=20):
        """
        从可用客户端中按样本数加权选择指定数量的客户端
        
        Args:
            round (int): 当前轮次，用于设置随机种子保证可重现性
            num_clients (int): 要选择的客户端数量，默认20
                               注意：实际选择数量是min(num_clients, 总客户端数)
        
        Returns:
            list: 选中的客户端对象列表
        """
        # 确保选择的客户端数量不超过总客户端数
        num_clients = min(num_clients, len(self.clients))
        
        # 设置随机种子，确保每轮比较时选择相同的客户端
        np.random.seed(round)
        
        # 均匀随机选择客户端（无替换采样）
        selected_clients = np.random.choice(self.clients, num_clients, replace=False)
        
        return selected_clients.tolist() # 返回选中的客户端列表

    def aggregate(self, wsolns):  
        """
        使用加权平均聚合客户端模型参数
        
        这是联邦学习中的核心聚合算法，通常称为FedAvg
        
        Args:
            wsolns (list): 包含(权重, 模型参数)元组的列表
                          权重通常是客户端的本地样本数
        
        Returns:
            list: 聚合后的平均模型参数
        """
        if not wsolns:
            return None
        
        total_weight = 0.0  # 总权重
        
        # 初始化累加器，与第一个解的参数结构相同
        _, first_soln = wsolns[0]
        if isinstance(first_soln, dict):
            # 如果参数是字典格式（state_dict）
            base = {k: torch.zeros_like(v, dtype=torch.float64) for k, v in first_soln.items()}
        else:
            # 如果参数是列表格式
            base = [torch.zeros_like(v, dtype=torch.float64) for v in first_soln]
        
        # 加权累加所有客户端的模型参数
        for (w, soln) in wsolns:  # w是本地样本数（权重）
            total_weight += w
            
            if isinstance(soln, dict):
                # 处理字典格式的参数
                for k, v in soln.items():
                    if isinstance(v, torch.Tensor):
                        base[k] += w * v.to(torch.float64)
                    else:
                        base[k] += w * torch.tensor(v, dtype=torch.float64)
            else:
                # 处理列表格式的参数
                for i, v in enumerate(soln):
                    if isinstance(v, torch.Tensor):
                        base[i] += w * v.to(torch.float64)
                    else:
                        base[i] += w * torch.tensor(v, dtype=torch.float64)

        # 计算加权平均
        if total_weight > 0:
            if isinstance(base, dict):
                averaged_soln = {k: v / total_weight for k, v in base.items()}
            else:
                averaged_soln = [v / total_weight for v in base]
        else:
            averaged_soln = base

        return averaged_soln