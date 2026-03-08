# d2l/__init__.py
"""
D2L (Dive into Deep Learning) 工具包

使用方法:
    from d2l import torch as d2l  # PyTorch 版本
"""

__version__ = '1.0.0'

# 导入子模块
from . import d2l_torch

# 导出模块列表
__all__ = ['d2l_torch.py']
