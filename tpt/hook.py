import os
import torch
import torch.nn as nn

from functools import partial

from .utils import process_dict_for_serialization, DataWriter, get_last_stack_no_torch
from .config import is_enable_save, get_save_dir, get_tensor_sample_size


if is_enable_save():
    save_path = os.path.join(get_save_dir(), f'PID-{os.getpid()}.tpt')
    if os.getenv('RANK', None) is not None and str(os.environ['RANK']).isdigit():
        save_path = os.path.join(get_save_dir(), f'RANK-{os.getenv("RANK")}.tpt')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    data_writer = DataWriter(save_path)


def forward_hook(name, module, args, kwargs, output):
    if is_enable_save():
        data = process_dict_for_serialization(
            {
                'args': args,
                'kwargs': kwargs,
                'output': output
            },
            # prefix=name,
            tensor_sample_size=get_tensor_sample_size()
        )
        data_writer.write(name, data, get_last_stack_no_torch())



def full_backward_hook(name, module, grad_input, grad_output):
    if is_enable_save():
        data = process_dict_for_serialization(
            {
                'grad_input': grad_input,
                'grad_output': grad_output,
            },
            # prefix=name,
            tensor_sample_size=get_tensor_sample_size()
        )
        data_writer.write(name, data, get_last_stack_no_torch())


# 1. 定义元类
class AutoHookMeta(type):
    depth = 0
    def __new__(mcs, name, bases, namespace):
        
        # 获取原始类中定义的 __init__ 方法
        original_init = namespace.get('__init__')
        
        # 2. 定义一个包装后的 __init__ 方法
        def wrapped_init(self, *args, **kwargs):
            mcs.depth += 1
            
            # 第一步：先安全地执行原始的 __init__，确保 nn.Module 底层状态初始化完毕
            if original_init:
                original_init(self, *args, **kwargs)
            else:
                # 如果子类没有定义 __init__，确保调用父类的 __init__
                super(type(self), self).__init__(*args, **kwargs)
            if mcs.depth == 1:
                prefix = f'{self.__class__.__name__}'
                # 第二步：在初始化完成后，执行全自动 Hook 注册逻辑
                for module_name, module in self.named_modules():
                    # 注册前向 Hook
                    module.register_forward_hook(
                        partial(forward_hook, f'{prefix}.{module_name}'), with_kwargs=True
                    )
                    module.register_full_backward_hook(
                        partial(full_backward_hook, f'{prefix}.{module_name}')
                    )
            mcs.depth -= 1
        
        # 3. 将包装后的 __init__ 注入到类的命名空间中
        namespace['__init__'] = wrapped_init
        
        # 4. 调用父类 type 的 __new__ 正式创建类
        object = super().__new__(mcs, name, bases, namespace)
        
        return object

# 5. 定义基类并指定元类
class BaseHookModule(nn.Module, metaclass=AutoHookMeta):
    def __init__(self):
        super().__init__()
