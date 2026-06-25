import argparse
import os
import numpy as np

from tpt.utils import logger
from tpt.utils import DataReader


def check_log(msg, is_error):
    if is_error:
        logger.error(f'{msg}')
    else:
        logger.info(f'{msg}')

def compare_listAtuple(key, v_a, v_b, warning_threshold=1e-5):
    v_a = np.array(v_a)
    v_b = np.array(v_b)
    if len(v_a) == len(v_b):
        rel_error = (np.abs(v_a - v_b) / (np.minimum(np.abs(v_a), np.abs(v_b)) + 1e-8))
        abs_error = np.abs(v_a - v_b)
        msg = f'[{key}] relative error: {rel_error.mean()} {rel_error.max()}, absolute error: {abs_error.mean()} {abs_error.max()}'
        check_log(msg, rel_error.mean() > warning_threshold)
    else:
        logger.error(f'[{key}] A_length: {len(v_a)}, B_length: {len(v_b)}')

def compare_common(key, v_a, v_b):
    msg = f'[{key}] A: {v_a}, B: {v_b}'
    check_log(msg, v_a != v_b)
        

def compare(argv=None):
    # 1. 创建解析器
    parser = argparse.ArgumentParser(description="用于对比两个数据文件的工具")

    # 2. 添加两个位置参数
    # 第一个参数：基准文件路径
    parser.add_argument("file_a", type=str, help="第一个数据文件的路径 (基准)")
    # 第二个参数：对比文件路径
    parser.add_argument("file_b", type=str, help="第二个数据文件的路径 (对比)")

    # 3. 解析命令行参数
    args = parser.parse_args(argv)

    # 4. 获取路径并进行基本校验
    path_a = args.file_a
    path_b = args.file_b

    if not os.path.exists(path_a):
        raise FileNotFoundError(f"文件不存在: {path_a}")
    if not os.path.exists(path_b):
        raise FileNotFoundError(f"文件不存在: {path_b}")

    logger.info(f"正在对比文件 A: {path_a}")
    logger.info(f"正在对比文件 B: {path_b}")
    
    data_a = DataReader(path_a)
    data_b = DataReader(path_b)
    for item_name_a, item_a in data_a.items():
        if item_name_a not in data_b:
            logger.warning(f'{item_name_a} not in B')
            continue
        logger.info(f'=========start compare {item_name_a}=========')
        item_b = data_b[item_name_a]
        for key_a, v_a in item_a.items():
            if key_a not in item_b:
                logger.warning(f'{key_a} not in B')
                continue
            v_b = item_b[key_a]
            if isinstance(v_a, (list, tuple)):
                compare_listAtuple(key_a, v_a, v_b)
            elif isinstance(v_a, (str, type(None), int, float, bool)):
                compare_common(key_a, v_a, v_b)
            else:
                logger.warning(f'not support compare type: {v_a.__class__.__name__},' + \
                    f'A: {v_a}, B: {v_b}')


if __name__ == "__main__":
    compare(['saves/PID-8573.tpt', 'saves/PID-7230.tpt'])
