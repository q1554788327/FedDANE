import numpy as np
import torch
from tqdm import trange, tqdm
from flearn.models.cifar10.jscc import Model
from .fedbase import BaseFedarated
from flearn.optimizer.pggd import PerturbedGradientDescent  


class Server(BaseFedarated):
    def __init__(self, params, learner, dataset):
        print('Using Federated Dane to Train')
        
        # 临时创建一个模型实例以获取模型参数
        temp_model = Model(20)
        
        # for name, param in temp_model.named_parameters():
        #     print(f"Model parameter: {name}, shape: {param.shape}, requires_grad: {param.requires_grad}")
        
        # 如果模型有state_dict()方法
        model_state = temp_model.state_dict().values()
        print(f"Model state dict length: {len(model_state)}")

        self.inner_opt = PerturbedGradientDescent(
            params=temp_model.parameters(),  # 传入模型的所有可训练参数
            lr=params['learning_rate'],   # 本地学习率
            mu=params['mu']               # 近端项系数，控制向全局模型收敛的强度
        )
        
        # 调用父类初始化，设置基本的联邦学习参数
        super(Server, self).__init__(params, learner, dataset)

    def train(self):
        '''使用FedDANE算法训练联邦学习模型'''
        print('Training with {} workers ---'.format(self.clients_per_round))
        
        # 主训练循环：进行指定轮数的联邦学习
        for i in trange(self.num_rounds, desc='Round: ', ncols=120):
            
            # ============ 模型评估阶段 ============
            # 每隔eval_every轮进行一次模型性能评估
            if i % self.eval_every == 0:
                stats = self.test()  # 在测试集上评估模型性能
                # stats_train = self.train_error_and_loss()  # 在训练集上评估模型性能
                
                ids, groups, psnrs = stats
                # 计算平均PSNR
                avg_psnr = np.mean(psnrs)
                tqdm.write('At round {} testing average PSNR: {:.2f}'.format(i, avg_psnr))

            # ============ 第一阶段：收集客户端梯度 ============
            # 随机选择参与梯度计算的客户端
            # selected_clients = self.select_clients(i, num_clients=self.clients_per_round)
            # 选择所有客户端参与计算
            selected_clients = self.clients
            
            cgrads = []  # 缓存客户端梯度的列表
            # 遍历选中的客户端，收集本地梯度
            for c in tqdm(selected_clients, desc='Grads: ', leave=False, ncols=120):
                # 将当前全局模型参数w^{t-1}发送给客户端
                c.set_params(self.global_model.state_dict())
                
                # 客户端计算本地梯度 ∇F_k(w^{t-1})
                grad, stats = c.solve_grad()
                cgrads.append(grad)
            
            # 聚合所有客户端梯度，计算平均梯度 g_t = avg(∇F_k(w^{t-1}))
            # 这是FedDANE算法的关键步骤：先收集梯度用于后续校正
            avg_gradient = self.aggregate(cgrads)

            # ============ 第二阶段：客户端本地训练 ============
            # 重新选择参与本地训练的客户端（可能与第一阶段不同）
            # selected_clients = self.select_clients(i, num_clients=self.clients_per_round)
            # 选择所有客户端参与训练
            selected_clients = self.clients

            csolns = []  # 缓存客户端训练结果的列表
            model_diffs = []  # 缓存客户端与全局模型的差异
            # 遍历选中的客户端进行本地训练
            for c in tqdm(selected_clients, desc='Solve: ', leave=False, ncols=120):
                
                # 1. 模型参数同步
                # 将全局模型参数w^{t-1}发送给客户端

                # 2. 设置FedDANE优化器的全局参数
                # 将全局模型参数设置为优化器的vstar（目标参数）
                c.optimizer.set_global_params(self.global_model.state_dict(), avg_gradient, c)

                # 4. 客户端本地训练
                # 使用DANE优化器进行多轮本地训练
                # 更新规则：w = w - lr*(∇F_k(w) + (g_avg - ∇F_k(w_global)) + μ*(w - w_global))
                soln = c.solve_inner(num_epochs=self.num_epochs, batch_size=self.batch_size)

                # 收集客户端训练后的模型参数
                csolns.append((c.num_samples, soln))
                
                # 收集客户端和全局模型的模型差异
        
            # ============ 第三阶段：模型聚合 ============
            # 聚合所有客户端的模型参数，更新全局模型w^t
            # 通常使用加权平均的方式聚合
            aggregate_params = self.aggregate(csolns)
            self.global_model.set_params(aggregate_params)
            
            # # 计算每个参数的平均模型差异
            # avg_diffs = []
            # for params in zip(*model_diffs):
            #     avg_diffs.append(torch.mean(torch.stack(params, dim=0), dim=0))

            # # 将平均模型差异应用到全局模型上
            # for param, diff in zip(self.global_model.parameters(), avg_diffs):
            #     param.data += diff

        # ============ 训练结束后的最终评估 ============
        # 在最终模型上进行性能评估
        stats = self.test()
        # stats_train = self.train_error_and_loss()
        
        # 输出最终的测试和训练准确率
        ids, groups, psnrs = stats
        # 计算平均PSNR
        avg_psnr = np.mean(psnrs)
        tqdm.write('At round {} testing average PSNR: {:.2f}'.format(self.num_rounds, avg_psnr))
        # tqdm.write('At round {} training accuracy: {}'.format(self.num_rounds, np.sum(stats_train[3])*1.0/np.sum(stats_train[2])))