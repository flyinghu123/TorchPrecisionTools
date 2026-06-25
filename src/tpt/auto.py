# -*- coding: utf-8 -*-
# This module is auto-imported via torch_precision_tools.pth on Python startup.

from tpt.config import initialize, is_enable_tpt


initialize()


if is_enable_tpt():
    import torch.nn as nn
    from tpt.hook import BaseHookModule
    from .utils import logger

    logger.info('ENABLE TPT AUTO HOOK')
    nn.Module = BaseHookModule