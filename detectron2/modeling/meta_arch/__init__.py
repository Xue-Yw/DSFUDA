# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.

from .build import META_ARCH_REGISTRY, build_model  # isort:skip

from .panoptic_fpn import PanopticFPN

# import all the meta_arch, so they will be registered
from .rcnn import GeneralizedRCNN, ProposalNetwork
from .retinanet import RetinaNet
from .semantic_seg import SEM_SEG_HEADS_REGISTRY, SemanticSegmentor, build_sem_seg_head
from .student_rcnn_clean_feature import student_RCNN_clean_feature
from .student_rcnn_clean import student_RCNN_clean
from .student_our_fpn import student_our_FPN
from .teacher_our_fpn import teacher_our_FPN
from .student_our_fpn_feature import student_our_FPN_feature


__all__ = list(globals().keys())
