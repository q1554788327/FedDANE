import numpy as np
import argparse
import importlib
import random
import os
import torch
from flearn.utils.model_utils import read_data

# ============ 全局配置参数 ============
# 支持的联邦学习优化算法列表
OPTIMIZERS = ['fedavg',    # 联邦平均算法
              'fedprox',   # 联邦近端算法  
              'feddane',   # DANE算法
              'fedddane',  # 分布式DANE算法
              'fedsgd']    # 联邦SGD算法

# 支持的数据集列表
DATASETS = ['sent140',           # 情感分析数据集
            'nist',              # NIST手写字符数据集(EMNIST)
            'shakespeare',       # 莎士比亚文本数据集
            'mnist',             # MNIST手写数字数据集
            'synthetic_iid',     # 合成IID数据集
            'synthetic_0_0',     # 合成非IID数据集(α=0, β=0)
            'synthetic_0.5_0.5', # 合成非IID数据集(α=0.5, β=0.5)
            'synthetic_1_1',     # 合成非IID数据集(α=1, β=1)
            'cifar10']           # CIFAR-10图像数据集

# 模型参数配置字典：每个数据集和模型组合对应的参数
MODEL_PARAMS = {
    # 情感分析任务模型参数
    'sent140.bag_dnn': (2,),                              # 词袋DNN：类别数
    'sent140.stacked_lstm': (25, 2, 100),                 # 堆叠LSTM：序列长度，类别数，隐藏单元数
    'sent140.stacked_lstm_no_embeddings': (25, 2, 100),   # 无嵌入层LSTM：序列长度，类别数，隐藏单元数
    
    # 字符识别任务模型参数
    'nist.mclr': (26,),        # 多分类逻辑回归：26个字母类别
    
    # 数字识别任务模型参数
    'mnist.mclr': (10,),       # 多分类逻辑回归：10个数字类别
    'mnist.cnn': (10,),        # 卷积神经网络：10个数字类别
    
    # 文本生成任务模型参数
    'shakespeare.stacked_lstm': (80, 80, 256),  # 堆叠LSTM：序列长度，嵌入维度，隐藏单元数
    
    # 合成数据集模型参数
    'synthetic.mclr': (10,),    # 多分类逻辑回归：10个类别
    
    # JSCC模型参数
    'cifar10.jscc': (20,),  # c=20，编码器输出通道数
}


def read_options():
    '''解析命令行参数或加载默认配置'''
    parser = argparse.ArgumentParser()

    # ============ 算法选择参数 ============
    parser.add_argument('--optimizer',
                        help='联邦学习优化算法名称',
                        type=str,
                        choices=OPTIMIZERS,
                        default='feddane')
    
    parser.add_argument('--dataset',
                        help='数据集名称',
                        type=str,
                        choices=DATASETS,
                        default='cifar10')
    
    parser.add_argument('--model',
                        help='模型名称（不含.py后缀）',
                        type=str,
                        default='jscc')  # 修正：去掉.py后缀

    # ============ 训练配置参数 ============
    parser.add_argument('--num_rounds',
                        help='联邦学习训练轮数',
                        type=int,
                        default=3)  # -1表示使用默认值
    
    parser.add_argument('--eval_every',
                        help='每隔多少轮进行一次模型评估',
                        type=int,
                        default=1)
    
    parser.add_argument('--clients_per_round',
                        help='每轮参与训练的客户端数量',
                        type=int,
                        default=3)

    # ============ 本地训练参数 ============
    parser.add_argument('--batch_size',
                        help='客户端本地训练的批大小',
                        type=int,
                        default=64)
    
    parser.add_argument('--num_epochs', 
                        help='客户端本地训练的轮数（epochs）',
                        type=int,
                        default=3)
    
    parser.add_argument('--num_iters',
                        help='客户端本地训练的迭代次数（iterations）',
                        type=int,
                        default=1)

    # ============ 优化器参数 ============
    parser.add_argument('--learning_rate',
                        help='本地训练学习率',
                        type=float,
                        default=0.0001)
    
    parser.add_argument('--mu',
                        help='FedProx/FedDANE算法的近端项系数',
                        type=float,
                        default=0)

    # ============ 系统配置参数 ============
    parser.add_argument('--seed',
                        help='随机种子，用于保证实验可重现',
                        type=int,
                        default=0)
    
    parser.add_argument('--drop_percent',
                        help='系统异构性：慢设备百分比',
                        type=float,
                        default=0.1)
    
    parser.add_argument('--device',
                    help='计算设备选择：cpu、cuda、cuda:0等',
                    type=str,
                    default='cuda:1',
                    choices=['auto', 'cpu', 'cuda', 'cuda:0', 'cuda:1', 'cuda:2', 'cuda:3'])


    # 解析命令行参数，处理可能的错误
    try: 
        parsed = vars(parser.parse_args())
    except IOError as msg: 
        parser.error(str(msg))

    # ============ 设置随机种子保证实验可重现 ============
    random.seed(1 + parsed['seed'])           # Python内置随机数生成器
    np.random.seed(12 + parsed['seed'])       # NumPy随机数生成器
    torch.manual_seed(123 + parsed['seed'])   # PyTorch随机数生成器
    
    # 如果使用GPU，同时设置CUDA随机种子
    if parsed['device'].startswith('cuda'):
        torch.cuda.manual_seed(123 + parsed['seed'])
        torch.cuda.manual_seed_all(123 + parsed['seed'])
        
    device = parsed['device']

    # ============ 动态加载模型类 ============
    # 构建模型模块路径
    if parsed['dataset'].startswith("synthetic"):  
        # 所有合成数据集使用相同的模型结构
        model_path = '%s.%s.%s.%s' % ('flearn', 'models', 'synthetic', parsed['model'])
    else:
        # 其他数据集使用各自专用的模型结构
        model_path = '%s.%s.%s.%s' % ('flearn', 'models', parsed['dataset'], parsed['model'])

    # 动态导入模型模块并获取Model类
    mod = importlib.import_module(model_path)
    learner = getattr(mod, 'Model')

    # ============ 动态加载训练器（优化算法）类 ============
    # 构建训练器模块路径
    opt_path = 'flearn.trainers.%s' % parsed['optimizer']
    # 动态导入训练器模块并获取Server类
    mod = importlib.import_module(opt_path)
    optimizer = getattr(mod, 'Server')

    # ============ 添加模型参数配置 ============
    # 根据数据集和模型组合获取对应的参数配置
    model_key = '.'.join(model_path.split('.')[2:])  # 提取'dataset.model'格式的键
    parsed['model_params'] = MODEL_PARAMS[model_key]

    # ============ 打印所有配置参数 ============
    maxLen = max([len(ii) for ii in parsed.keys()])  # 找到最长的参数名长度
    fmtString = '\t%' + str(maxLen) + 's : %s'        # 格式化字符串
    print('Arguments:')
    # 按字母顺序打印所有参数及其值
    for keyPair in sorted(parsed.items()): 
        print(fmtString % keyPair)

    return parsed, learner, optimizer


def main():
    '''主函数：联邦学习实验入口'''
    
    # ============ 解析配置参数 ============
    options, learner, optimizer = read_options()

    # ============ 加载数据集 ============
    # 构建训练集和测试集的文件路径
    train_path = os.path.join('data', options['dataset'], 'data', 'train')
    test_path = os.path.join('data', options['dataset'], 'data', 'test')
    
    # 读取并预处理数据集（包含客户端数据分割）
    dataset = read_data(train_path, test_path)

    # ============ 实例化并启动联邦学习训练器 ============
    # 创建训练器实例：传入配置、模型类和数据集
    t = optimizer(options, learner, dataset)
    
    # 开始联邦学习训练过程
    t.train()


if __name__ == '__main__':
    main()