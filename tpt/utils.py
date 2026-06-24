import os
import inspect
import json
import numbers
import torch
import site

from collections import defaultdict
from pathlib import Path


class DataWriter(object):
    def __init__(self, file_path):
        self.file_handle = open(file_path, mode='w')
    
    def write(self, key, data, stack):
        self.file_handle.write(f'{key}, {json.dumps(data)}, {stack}\n')
        self.file_handle.flush()
    
    def close(self):
        self.file_handle.close()


class DataReader(dict):
    def __init__(self, file_path):
        super().__init__()
        data = defaultdict(list)
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                key, value = line.split(', ', 1)
                value, stack = value.rsplit(', ', 1)
                data[key].append(value)
        self.update(data)


def process_dict_for_serialization(data, tensor_sample_size=10, prefix=''):
    """
    递归处理字典：
    1. 如果是 torch.Tensor：进行采样，转换为原生 list/dict。
    2. 如果是其他可 JSON 序列化类型：保留原样。
    3. 如果是不可 JSON 序列化类型：转换为 str。
    """
    ret = {}
    
    # 1. 处理 torch.Tensor 类型
    if isinstance(data, torch.Tensor):
        # 第一步：采样（这里以取前 tensor_sample_size 个元素为例，可根据需求改为随机采样）
        # 先将 tensor 展平为一维，方便采样
        flat_data = data.detach().cpu().to(torch.float32).flatten()
        if flat_data.numel() > tensor_sample_size:
            sampled_data = flat_data[torch.randperm(flat_data.numel())[:tensor_sample_size]]
        else:
            sampled_data = flat_data
        
        # 第二步：转换为 Python 原生类型（CPU + tolist() 能完美转为 list 或 标量）
        ret[prefix] = sampled_data.tolist()

    # 2. 处理字典类型（递归遍历）
    elif isinstance(data, dict):
        for k, v in data.items():
            ret.update(process_dict_for_serialization(v, tensor_sample_size, prefix=f'{prefix}.{k}' if prefix else f'{k}'))
    
    # 3. 处理列表或元组类型（递归遍历）
    elif isinstance(data, (list, tuple)) and not all([isinstance(v, numbers.Number) for v in data]):
        for idx, v in enumerate(data):
            ret.update(process_dict_for_serialization(v, tensor_sample_size, prefix=f'{prefix}.{idx}' if prefix else f'{idx}'))

    # 4. 处理其他类型：尝试 JSON 序列化，如果失败则转为 str
    else:
        try:
            # json.dumps 会检查对象是否可序列化，如果不可序列化会抛出 TypeError
            json.dumps(data)
            ret[prefix] = data
        except (TypeError, OverflowError):
            ret[prefix] = str(data)
    
    return ret


def is_in_directory(fn, target_dir):
    abs_fn = Path(fn).resolve()
    abs_dir = Path(target_dir).resolve()
    return abs_fn.is_relative_to(abs_dir)


def get_last_stack_no_torch():
    for stack in inspect.stack()[2:]:
        filename = stack.filename
        lineno = stack.lineno
        if is_in_directory(filename, os.path.join(site.getsitepackages()[0], 'torch')):
            continue
        return f'{filename}:{lineno}'