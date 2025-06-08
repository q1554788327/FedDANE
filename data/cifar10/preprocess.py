import os
import json
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split

def create_cifar10_federated_data(num_clients=10, alpha=0.5, save_dir='/mnt/7t/tz/github/uav-jscc-fl/FedDANE/data/cifar10/data'):
    """
    创建CIFAR10的联邦学习数据分割
    
    Args:
        num_clients: 客户端数量
        alpha: Dirichlet分布参数，控制数据异构程度
        save_dir: 保存目录
    """
    
    # 下载CIFAR10数据集并进行归一化（像素值/255）
    transform = transforms.Compose([
        transforms.ToTensor(),  # ToTensor会自动将像素值从[0,255]缩放到[0.0,1.0]
    ])

    # 加载数据集并应用转换
    trainset = torchvision.datasets.CIFAR10(
        root='/mnt/7t/tz/github/uav-jscc-fl/FedDANE/data/cifar10', 
        train=True, 
        download=True, 
        transform=transform
    )
    testset = torchvision.datasets.CIFAR10(
        root='/mnt/7t/tz/github/uav-jscc-fl/FedDANE/data/cifar10', 
        train=False, 
        download=True, 
        transform=transform
    )

    # 每个类采样20条数据
    def sample_data(dataset, num_per_class=20):
        """采样数据集并应用归一化"""
        data = []
        labels = []
        
        # 遍历每个类别
        for class_id in range(10):
            # 获取该类别的所有样本索引
            class_indices = [i for i, (_, label) in enumerate(dataset) if label == class_id]
            # 随机选择指定数量的样本
            selected_indices = np.random.choice(class_indices, num_per_class, replace=False)
            
            # 收集样本和标签
            for idx in selected_indices:
                img, label = dataset[idx]  # 通过索引访问，自动应用transform
                data.append(img.numpy())
                labels.append(label)
        
        return np.array(data), np.array(labels)

    # 采样训练集和测试集（已归一化）
    train_data, train_labels = sample_data(trainset)
    test_data, test_labels = sample_data(testset)
    
    # 创建非IID数据分割
    train_client_data = create_noniid_split(train_data, train_labels, 
                                           num_clients, alpha)
    test_client_data = create_noniid_split(test_data, test_labels, 
                                          num_clients, alpha)
    
    # 保存数据
    os.makedirs(f'{save_dir}/train', exist_ok=True)
    os.makedirs(f'{save_dir}/test', exist_ok=True)
    
    # 保存训练数据
    for i in range(num_clients):
        train_file = f'{save_dir}/train/client_{i}.json'
        with open(train_file, 'w') as f:
            json.dump({
                'users': [f'client_{i}'],
                'user_data': {
                    f'client_{i}': {
                        'x': train_client_data[i]['x'].tolist(),  # 输入图像
                        'y': train_client_data[i]['x'].tolist()   # 目标图像（与输入相同，用于重建任务）
                    }
                },
                'num_samples': [len(train_client_data[i]['x'])]
            }, f)
    
    # 保存测试数据  
    for i in range(num_clients):
        test_file = f'{save_dir}/test/client_{i}.json'
        with open(test_file, 'w') as f:
            json.dump({
                'users': [f'client_{i}'],
                'user_data': {
                    f'client_{i}': {
                        'x': test_client_data[i]['x'].tolist(),   # 输入图像
                        'y': test_client_data[i]['x'].tolist()    # 目标图像（与输入相同）
                    }
                },
                'num_samples': [len(test_client_data[i]['x'])]
            }, f)

def create_noniid_split(data, labels, num_clients, alpha):
    """使用Dirichlet分布创建非IID数据分割"""
    num_classes = len(np.unique(labels))
    client_data = []
    
    # 为每个类别创建Dirichlet分布
    for k in range(num_classes):
        idx_k = np.where(labels == k)[0]
        np.random.shuffle(idx_k)
        
        # Dirichlet分布采样
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)
        proportions[-1] = len(idx_k)
        
        # 分割数据
        start_idx = 0
        for i in range(num_clients):
            if i >= len(client_data):
                client_data.append({'x': [], 'y': []})
            
            end_idx = proportions[i]
            client_indices = idx_k[start_idx:end_idx]
            
            if len(client_data[i]['x']) == 0:
                client_data[i]['x'] = data[client_indices]
                client_data[i]['y'] = labels[client_indices]
            else:
                client_data[i]['x'] = np.concatenate([client_data[i]['x'], 
                                                     data[client_indices]])
                client_data[i]['y'] = np.concatenate([client_data[i]['y'], 
                                                     labels[client_indices]])
            
            start_idx = end_idx
    
    return client_data

if __name__ == '__main__':
    create_cifar10_federated_data(num_clients=10, alpha=0.5)