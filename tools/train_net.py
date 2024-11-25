#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
"""
A main training script.

This scripts reads a given config file and runs the training or evaluation.
It is an entry point that is made to train standard models in detectron2.

In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use detectron2 as an library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.
"""

import logging
import os
from collections import OrderedDict
import torch
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    verify_results,
    CityscapeDetectionEvaluator
)
from detectron2.modeling import GeneralizedRCNNWithTTA

# 加的验证
from detectron2.engine import HookBase
from detectron2.data import build_detection_train_loader
import detectron2.utils.comm as comm



#os.environ['CUDA_VISIBLE_DEVICES']='0'

# 构建评估器
def build_evaluator(cfg, dataset_name, output_folder=None):
    """
    Create evaluator(s) for a given dataset.
    This uses the special metadata "evaluator_type" associated with each builtin dataset.
    For your own dataset, you can simply create an evaluator manually in your
    script and do not have to worry about the hacky if-else logic here.
    """
    if output_folder is None:
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")#读取保存地址
    evaluator_list = []
    evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type#根据数据集类型，配置不同的评估器
    if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
        evaluator_list.append(
            SemSegEvaluator(
                dataset_name,
                distributed=True,
                output_dir=output_folder,
            )
        )
    if evaluator_type in ["coco", "coco_panoptic_seg"]:
        evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
    if evaluator_type == "coco_panoptic_seg":
        evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
    if evaluator_type == "cityscapes_instance":
        assert (
            torch.cuda.device_count() > comm.get_rank()
        ), "CityscapesEvaluator currently do not work with multiple machines."
        return CityscapesInstanceEvaluator(dataset_name)
    if evaluator_type == "cityscapes_sem_seg":
        assert (
            torch.cuda.device_count() > comm.get_rank()
        ), "CityscapesEvaluator currently do not work with multiple machines."
        return CityscapesSemSegEvaluator(dataset_name)
    elif evaluator_type == "pascal_voc":
        return PascalVOCDetectionEvaluator(dataset_name)
    elif evaluator_type == "lvis":
        return LVISEvaluator(dataset_name, output_dir=output_folder)
    if evaluator_type == "cityscape":
        return CityscapeDetectionEvaluator(dataset_name)
    if len(evaluator_list) == 0:
        raise NotImplementedError(
            "no Evaluator for the dataset {} with the type {}".format(dataset_name, evaluator_type)
        )
    elif len(evaluator_list) == 1:
        return evaluator_list[0]
    return DatasetEvaluators(evaluator_list)

#训练
class Trainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains pre-defined default logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can write your
    own training loop. You can use "tools/plain_train_net.py" as an example.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        return build_evaluator(cfg, dataset_name, output_folder)

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA
        # Only support some R-CNN models.
        logger.info("Running inference with test-time augmentation ...")
        model = GeneralizedRCNNWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res

def add_new_config(cfg):
	cfg.MODEL.RPN.ANCHOR_SIZES= True  # 新增加的变量temp


#读取命令行参数
def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()#get_cfg函数负责返回detectron2的默认参数的一份拷贝，而这份参数是以CfgNode进行存储的，包含了大量的网络信息，但是要注意的是缺少了例如权重路径之类的关键信息，因此需要进行设置
    
    add_new_config(cfg)

    cfg.merge_from_file(args.config_file)#是CfgNode的类方法，他会进行参数更新
    cfg.merge_from_list(args.opts)
    cfg.freeze()#使所有的参数不可变，然后返回这个设置好的固定参数组
    default_setup(cfg, args)#输出了一些配置信息，环境，配置文件名什么的
    return cfg




import os
from tqdm import tqdm
import time

# declare which gpu device to use
cuda_device = '0'

def check_mem(cuda_device):
    devices_info = os.popen('"/usr/bin/nvidia-smi" --query-gpu=memory.total,memory.used --format=csv,nounits,noheader').read().strip().split("\n")
    total, used = devices_info[int(cuda_device)].split(',')
    return total,used

def occumpy_mem(cuda_device):
    total, used = check_mem(cuda_device)
    total = int(total)
    used = int(used)
    max_mem = int(total * 0.9)
    block_mem = max_mem - used
    x = torch.cuda.FloatTensor(256,1024,block_mem)
    del x
    
# 增加的验证
class ValidationLoss(HookBase):
    def __init__(self, cfg, DATASETS_VAL_NAME):#多加一个DATASETS_VAL_NAME参数（小改动）
        super().__init__()
        self.cfg = cfg.clone()
        self.cfg.DATASETS.TRAIN = DATASETS_VAL_NAME##
        self._loader = iter(build_detection_train_loader(self.cfg))
        
    def after_step(self):
        data = next(self._loader)
        with torch.no_grad():
            loss_dict = self.trainer.model(data)
            
            losses = sum(loss_dict.values())
            assert torch.isfinite(losses).all(), loss_dict

            loss_dict_reduced = {"val_" + k: v.item() for k, v in 
                                 comm.reduce_dict(loss_dict).items()}
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            if comm.is_main_process():
                self.trainer.storage.put_scalars(total_val_loss=losses_reduced, 
                                                 **loss_dict_reduced)

# 更改预训练模型中的键
def modify_pretrained_weights(pretrained_weights, rename_mapping):
    """
    Modify the keys in the pretrained weights according to the rename_mapping.
    Args:
        pretrained_weights: A dictionary containing the pretrained weights.
        rename_mapping: A dictionary where the keys are the old names and the values are the new names.
    Returns:
        A new dictionary with the keys renamed.
    """
    from collections import OrderedDict
    modified_weights = OrderedDict()
    for old_key, weight in pretrained_weights.items():
        new_key = old_key
        for old_prefix, new_prefix in rename_mapping.items():
            if old_key.startswith(old_prefix):
                new_key = new_key.replace(old_prefix, new_prefix, 1)
                break
        modified_weights[new_key] = weight
    return modified_weights

def main(args):
    # import pdb
    # pdb.set_trace()
    cfg = setup(args)#自定义函数，读取命令行参数

    if args.eval_only:#评估模式
        model = Trainer.build_model(cfg)#自定义函数，根据配置构建模型
        
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )#加载模型权重

        res = Trainer.test(cfg, model)#自定义函数，使用配置和模型进行测试
        if cfg.TEST.AUG.ENABLED:#检查是否启用测试时的数据增强
            res.update(Trainer.test_with_TTA(cfg, model))#更新结果
        if comm.is_main_process():#检查是否是主进程
            verify_results(cfg, res)#在主进程中验证测试结果
        return res

    """
    If you'd like to do anything fancier than the standard training logic,
    consider writing your own training loop (see plain_train_net.py) or
    subclassing the trainer.
    """
    # 增加验证集
    # val_loss = ValidationLoss(cfg, cfg.DATASETS.TEST)  ##多加的参数
    # trainer.register_hooks([val_loss])
    # # swap the order of PeriodicWriter and ValidationLoss
    # trainer._hooks = trainer._hooks[:-2] + trainer._hooks[-2:][::-1]

    #  
    # 
    # 重命名权重键
    # 加载预训练模型的 state_dict

    # 
    # 
    # 修改权重
    # if not args.resume:  # 仅在不是恢复训练时加载预训练权重
    #     print("Loading and modifying pretrained weights...")
    #     # import pdb
    #     # pdb.set_trace()
    #     pretrained_weights = torch.load(cfg.MODEL.WEIGHTS)

    #     # 如果不包含 "model" 键，重新包装权重
    #     if "model" not in pretrained_weights:
    #         pretrained_weights = {"model": pretrained_weights}

    #     # 定义键名修改的映射关系
    #     rename_mapping = {
    #         "backbone.bottom_up.res5.":"roi_heads.res5.",
    #         "backbone.bottom_up.": "backbone."
    #         # 添加更多的映射规则，如果需要
    #     }
        
    #     # import pdb
    #     # pdb.set_trace()

    #     # 修改预训练权重的键名
    #     modified_weights = modify_pretrained_weights(pretrained_weights['model'], rename_mapping)
    #     # modified_weights = modify_pretrained_weights(pretrained_weights, rename_mapping)

    #     # pth
    #     # trainer.model.load_state_dict(modified_weights)

    #     # # 将修改后的权重重新保存到新文件中
    #     new_weights_path = os.path.join(cfg.OUTPUT_DIR, "modified_weights.pth")
    #     torch.save({"model": modified_weights}, new_weights_path)

    #     # import pdb
    #     # pdb.set_trace()
    #     cfg.defrost()
    #     cfg.MODEL.WEIGHTS = new_weights_path
    #     cfg.freeze()

        # 将修改后的权重加载到模型中
        # checkpointer = DetectionCheckpointer(trainer.model, save_dir=cfg.OUTPUT_DIR)
        # checkpointer.resume_or_load(new_weights_path, resume=args.resume)
        # 
        # 
        # 
    
    trainer = Trainer(cfg)#训练模式 
    trainer.resume_or_load(resume=args.resume)#复训练状态或加载初始权重 False:加载初始权重
    if cfg.TEST.AUG.ENABLED:#检查是否启用训练时的数据增强
        trainer.register_hooks(
            [hooks.EvalHook(0, lambda: trainer.test_with_TTA(cfg, trainer.model))]
        )#注册一个钩子，在训练期间进行带 TTA 的测试。
    
    return trainer.train()#启动训练并返回训练结果


if __name__ == "__main__":
    
    args = default_argument_parser().parse_args()#解析命令行参数
    print("Command Line Args:", args)
    launch(#通过 launch 函数启动主程序 main，并传递一些关于分布式训练的参数
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
