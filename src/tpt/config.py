import os


# 是否开启TPT
_ENABLE_TPT = False

# 保存位置
_SAVE_DIR = None

# tensor采样数量
_TENSOR_SAMPLE_SIZE = 10


def is_enable_tpt():
    return _ENABLE_TPT

def is_enable_save():
    return _SAVE_DIR is not None

def get_save_dir():
    return _SAVE_DIR

def get_tensor_sample_size():
    return _TENSOR_SAMPLE_SIZE

def initialize():
    if str(os.getenv('ENABLE_TPT', None)).lower() in ['true', '1', 'on']:
        global _ENABLE_TPT
        _ENABLE_TPT = True

    if os.getenv('TPT_SAVE_DIR', None) is not None:
        global _SAVE_DIR
        _SAVE_DIR = os.environ['TPT_SAVE_DIR']
    
    if os.getenv('TPT_TENSOR_SAMPLE_SIZE', None) is not None:
        global _TENSOR_SAMPLE_SIZE
        _TENSOR_SAMPLE_SIZE = int(os.environ['TPT_TENSOR_SAMPLE_SIZE'])
