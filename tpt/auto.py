# -*- coding: utf-8 -*-
# This module is auto-imported via torch_precision_tools.pth on Python startup.

import os


if str(os.getenv('ENABLE_TPT', None)).lower() in ['true', '1', 'on']:
    print('ENABLE TPT AUTO HOOK')
    import torch.nn as nn
    from tpt.base import BaseModule
    nn.Module = BaseModule