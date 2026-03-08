# d2l_utils.py
"""
D2L (Dive into Deep Learning) 常用工具函数集合
"""
import time
import random
import numpy as np
import os
import torch
import torchvision

from torch.utils import data
from torchvision import transforms
from matplotlib import pyplot as plt
from matplotlib_inline import backend_inline

# ==================== 数据处理 ====================

def synthetic_data(w, b, num_examples):
    """生成数据集(y=Xw+b+噪声)"""
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))


def data_iter(batch_size, features, labels):
    """数据迭代器（从零开始实现）"""
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(
            indices[i: min(i + batch_size, num_examples)]
        )
        yield features[batch_indices], labels[batch_indices]


def load_array(data_arrays, batch_size, is_train=True):
    """构造一个PyTorch数据迭代器"""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train, num_workers=0)


# ==================== 模型相关 ====================

def linreg(X, w, b):
    """线性回归模型"""
    return torch.matmul(X, w) + b


def squared_loss(y_hat, y):
    """均方损失"""
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2


def sgd(params, lr, batch_size):
    """小批量随机梯度下降"""
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()


# ==================== 可视化 ====================

def use_svg_display():
    """使用svg格式在Jupyter中显示绘图"""
    backend_inline.set_matplotlib_formats('svg')


def set_figsize(figsize=(3.5, 2.5)):
    """设置matplotlib的图表大小"""
    use_svg_display()
    plt.rcParams['figure.figsize'] = figsize


def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):
    """绘制图像列表"""
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            ax.imshow(img.numpy())
        else:
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes


# ==================== 工具类 ====================

class Timer:
    """记录多次运行时间"""
    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        """启动计时器"""
        self.tik = time.time()

    def stop(self):
        """停止计时器并将时间记录在列表中"""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """返回平均时间"""
        return sum(self.times) / len(self.times)

    def sum(self):
        """返回时间总和"""
        return sum(self.times)

    def cumsum(self):
        """返回累计时间"""
        return np.array(self.times).cumsum().tolist()


def get_fashion_mnist_labels(labels):
    """返回Fashion-MNIST数据集的文本标签"""
    text_labels = [
        't-shirt', 'trouser', 'pullover', 'dress', 'coat',
        'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot'
    ]
    return [text_labels[int(i)] for i in labels]


def get_dataloader_workers():
    """使用4个进程来读取数据"""
    return 4


def load_data_fashion_mnist(batch_size, resize=None):
    """下载Fashion-MNIST数据集，然后将其加载到内存中"""
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)

    data_path = "../data/FashionMNIST/raw"

    # FashionMNIST数据集的关键文件
    required_train_files = [
        "train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte.gz"
    ]

    required_test_files = [
        "t10k-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz"
    ]

    # 检查所有必需文件是否存在
    train_files_exist = all(
        os.path.exists(os.path.join(data_path, f)) for f in required_train_files
    )

    if train_files_exist:
        print("训练数据集已存在，直接读取...")
        download_train_needed = False
    else:
        print("训练数据集不存在或不完整，正在下载...")
        download_train_needed = True

    test_files_exist = all(
        os.path.exists(os.path.join(data_path, f)) for f in required_test_files
    )

    if test_files_exist:
        print("测试数据集已存在，直接读取...")
        download_test_needed = False
    else:
        print("测试数据集不存在或不完整，正在下载...")
        download_test_needed = True

    mnist_train = torchvision.datasets.FashionMNIST(
        root="../data", train=True, transform=trans, download=download_train_needed
    )
    mnist_test = torchvision.datasets.FashionMNIST(
        root="../data", train=False, transform=trans, download=download_test_needed
    )

    return (data.DataLoader(mnist_train, batch_size, shuffle=True, num_workers=get_dataloader_workers()),
            data.DataLoader(mnist_test, batch_size, shuffle=False, num_workers=get_dataloader_workers()))

def accuracy(y_hat, y):
    """计算预测正确的数量"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        # 如果y_hat是二维的概率分布，
        # 例如y_hat = [[0.1, 0.3, 0.6],
        #               [0.3, 0.2, 0.5]]
        # 取每一行最大值的索引，y_hat.argmax(1) = [2, 2]
        y_hat = y_hat.argmax(1)
    # 先将y_hat的值转换类型，与y的数据类型一致
    # y_hat = y_hat.type(y.dtype)
    cmp = y_hat.type(y.dtype) == y
    # 运算之后的cmp的数据类型为布尔类型 cmp = [True, False, False, ..., True]
    print(cmp)
    # 先将cmp的布尔类型转换成int类型: 0或1，然后再求和，求和之后再转换成float类型，方便后续计算精确度
    return float(cmp.type(y.dtype).sum())