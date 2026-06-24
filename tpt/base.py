import torch
import torch.nn as nn

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
                # 第二步：在初始化完成后，执行全自动 Hook 注册逻辑
                for module_name, module in self.named_modules():
                    # 注册前向 Hook
                    module.register_forward_hook(
                        lambda m, inp, out: print(f"Conv2d '{module_name}' 被触发，输出形状: {out.shape}")
                    )
            mcs.depth -= 1
        
        # 3. 将包装后的 __init__ 注入到类的命名空间中
        namespace['__init__'] = wrapped_init
        
        # 4. 调用父类 type 的 __new__ 正式创建类
        object = super().__new__(mcs, name, bases, namespace)
        
        return object

# 5. 定义基类并指定元类
class BaseModule(nn.Module, metaclass=AutoHookMeta):
    def __init__(self):
        super().__init__()
