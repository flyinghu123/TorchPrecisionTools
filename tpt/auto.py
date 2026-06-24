# -*- coding: utf-8 -*-
# This module is auto-imported via torch_precision_tools.pth on Python startup.

from .config import initialize, is_enable_tpt


initialize()


if is_enable_tpt():
    print('ENABLE TPT AUTO HOOK')
    import torch.nn as nn
    from tpt.hook import BaseHookModule
    nn.Module = BaseHookModule