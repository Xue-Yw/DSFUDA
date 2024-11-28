# Copyright (c) Facebook, Inc. and its affiliates.
import torch

from detectron2.utils.logger import _log_api_usage
from detectron2.utils.registry import Registry

#加载了META_ARCH这个容器
META_ARCH_REGISTRY = Registry("META_ARCH")  # noqa F401 isort:skip
META_ARCH_REGISTRY.__doc__ = """
Registry for meta-architectures, i.e. the whole model.

The registered object will be called with `obj(cfg)`
and expected to return a `nn.Module` object.
"""


def build_model(cfg):
    """
    Build the whole model architecture, defined by ``cfg.MODEL.META_ARCHITECTURE``.
    Note that it does not load any weights from ``cfg``.
    """
    meta_arch = cfg.MODEL.META_ARCHITECTURE#获取模型的元架构，元架构描述了模型的整体结构,在配置文件中定义
    model = META_ARCH_REGISTRY.get(meta_arch)(cfg)#使用元架构注册表（META_ARCH_REGISTRY）中注册的构建函数来构建模型。通过 meta_arch 获取对应的构建函数，然后将配置对象 cfg 传递给这个构建函数，得到模型对象 model
    model.to(torch.device(cfg.MODEL.DEVICE))#模型移动到配置中指定的设备上
    _log_api_usage("modeling.meta_arch." + meta_arch)
    return model
