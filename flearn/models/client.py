import numpy as np
import torch

class Client(object):
    
    def __init__(self, id, group=None, train_data={'x':[],'y':[]}, eval_data={'x':[],'y':[]}, model=None, optimizer=None):
        self.model = model
        self.id = id
        self.group = group
        
        # 客户端优化器
        self.optimizer = optimizer
        
        # 存储原始数据
        self._train_data_np = {k: np.array(v) for k, v in train_data.items()}
        self._eval_data_np = {k: np.array(v) for k, v in eval_data.items()}
        
        # 初始化tensor数据（CPU）
        self.train_data = None
        self.eval_data = None
        self._current_device = None
        
        # 首次同步设备
        self._sync_data_device()
        
        self.num_samples = len(self._train_data_np['y'])
        self.test_samples = len(self._eval_data_np['y'])

    def _get_model_device(self):
        """获取模型所在设备"""
        if self.model is not None:
            try:
                return next(self.model.parameters()).device
            except StopIteration:
                return torch.device('cpu')
        return torch.device('cpu')

    def _sync_data_device(self):
        """同步数据到模型设备"""
        model_device = self._get_model_device()
        
        # 如果设备没有变化，不需要重新移动数据
        if self._current_device == model_device and self.train_data is not None:
            return
        
        print(f"客户端 {self.id} 将数据移动到设备: {model_device}")
        
        # 转换并移动训练数据
        self.train_data = {}
        for k, v in self._train_data_np.items():
            if k in ['x', 'y']:  # 对于JSCC，x和y都是图像数据
                tensor = torch.tensor(v, dtype=torch.float32)
            else:
                tensor = torch.tensor(v)
            self.train_data[k] = tensor.to(model_device)
        
        # 转换并移动测试数据
        self.eval_data = {}
        for k, v in self._eval_data_np.items():
            if k in ['x', 'y']:  # 对于JSCC，x和y都是图像数据
                tensor = torch.tensor(v, dtype=torch.float32)
            else:
                tensor = torch.tensor(v)
            self.eval_data[k] = tensor.to(model_device)
        
        # 如果y是标签但我们需要图像，使用x作为目标
        if self.train_data['y'].dim() == 1:  # 检查是否是1D标签
            print(f"客户端 {self.id}: 检测到标签数据，将使用输入图像作为重建目标")
            self.train_data['y'] = self.train_data['x'].clone()
            self.eval_data['y'] = self.eval_data['x'].clone()
        
        self._current_device = model_device

    def set_params(self, model_params):
        '''set model parameters'''
        self.model.set_params(model_params)
        # 模型参数更新后，检查是否需要同步数据设备
        self._sync_data_device()

    def get_grads(self):
        '''get model gradient'''
        self._sync_data_device()  # 确保数据在正确设备上
        return self.model.get_gradients()

    def test(self):
        '''tests current model on local eval_data'''
        self._sync_data_device()  # 确保数据在正确设备上
        psnr = self.model.test(self.eval_data)
        return psnr
    
    def train_error_and_loss(self):
        '''returns training error and loss on local train_data'''
        self._sync_data_device()
        psnr, loss = self.model.train_error_and_loss(self.train_data)
        return psnr, loss

    def solve_inner(self, num_epochs=1, batch_size=10):
        '''Solves local optimization problem'''
        self._sync_data_device()  # 确保数据在正确设备上
        bytes_w = self.model.size
        soln= self.model.solve_inner(self.train_data, self.optimizer, num_epochs, batch_size)
        bytes_r = self.model.size
        return soln
    
    def solve_iters(self, num_iters=1, batch_size=10):
        '''Solves local optimization problem

        Return:
            1: num_samples: number of samples used in training
            1: soln: local optimization solution
            2: bytes read: number of bytes received
            2: comp: number of FLOPs executed in training process
            2: bytes_write: number of bytes transmitted
        '''

        bytes_w = self.model.size
        soln, comp = self.model.solve_iters(self.train_data, self.optimizer, num_iters, batch_size)
        bytes_r = self.model.size
        return (self.num_samples, soln), (bytes_w, comp, bytes_r)
    
    def solve_grad(self):
        '''get model gradient with cost'''
        bytes_w = self.model.size
        grads = self.model.cal_gradients(self.train_data, self.optimizer)
        comp = self.model.flops * self.num_samples
        bytes_r = self.model.size
        return ((self.num_samples, grads), (bytes_w, comp, bytes_r))