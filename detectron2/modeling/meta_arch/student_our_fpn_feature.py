
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
import torch
from torch import nn
import matplotlib.pyplot as plt
import seaborn as sns
import random
import os


from detectron2.config import configurable
from detectron2.data.detection_utils import convert_image_to_rgb
from detectron2.structures import ImageList, Instances
from detectron2.utils.events import get_event_storage
from detectron2.utils.logger import log_first_n

from ..backbone import Backbone, build_backbone
from ..postprocessing import detector_postprocess
from ..proposal_generator import build_proposal_generator
from ..roi_heads import build_roi_heads
from .build import META_ARCH_REGISTRY

import pdb
import cv2
import torch.nn.functional as F
from matplotlib import pyplot as plt
from .losses import GraphConLoss
from .GCN import GCN

from .diffkd_modules import DiffusionModel, NoiseAdapter, AutoEncoder, DDIMPipeline
from .scheduling_ddim import DDIMScheduler


import sys
from torchvision import transforms


__all__ = ["student_our_FPN_feature"]

@META_ARCH_REGISTRY.register()
class student_our_FPN_feature(nn.Module):
    """
    student_our_FPN R-CNN. Any models that contains the following three components:
    1. Per-image feature extraction (aka backbone)
    2. Region proposal generation
    3. Per-region feature extraction and prediction
    """

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        proposal_generator: nn.Module,
        roi_heads: nn.Module,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        input_format: Optional[str] = None,
        vis_period: int = 0,
        # 
        # 
        # 
        student_channels=256,
        teacher_channels=256,
        kernel_size=3,
        inference_steps=5,
        num_train_timesteps=1000,
        use_ae=False,
        ae_channels=None,
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            proposal_generator: a module that generates proposals using backbone features
            roi_heads: a ROI head that performs per-region computation
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
            input_format: describe the meaning of channels of input. Needed by visualization
            vis_period: the period to run visualization. Set to 0 to disable.
        """
        super().__init__()
        self.backbone = backbone
        self.proposal_generator = proposal_generator
        self.roi_heads = roi_heads

        self.input_format = input_format
        self.vis_period = vis_period
        if vis_period > 0:
            assert input_format is not None, "input_format is required for visualization!"

        self.register_buffer("pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.tensor(pixel_std).view(-1, 1, 1), False)
        assert (
            self.pixel_mean.shape == self.pixel_std.shape
        ), f"{self.pixel_mean} and {self.pixel_std} have different shapes!"
 
        
        self.t_s_loss = nn.MSELoss() 
        self.use_ae = use_ae
        self.diffusion_inference_steps = inference_steps
        # AE for compress teacher feature
        if use_ae:
            if ae_channels is None:
                ae_channels = teacher_channels // 2
            self.ae = AutoEncoder(teacher_channels, ae_channels)
            teacher_channels = ae_channels
        
        # transform student feature to the same dimension as teacher
        self.trans = nn.Conv2d(student_channels, teacher_channels, 1)
        self.diffusionmodel = DiffusionModel(channels_in=teacher_channels, kernel_size=kernel_size)
        self.scheduler = DDIMScheduler(num_train_timesteps=num_train_timesteps, clip_sample=False, beta_schedule="linear")
        self.noise_adapter = NoiseAdapter(teacher_channels, kernel_size)
        self.pipeline = DDIMPipeline(self.diffusionmodel, self.scheduler, self.noise_adapter)
        self.proj = nn.Sequential(nn.Conv2d(teacher_channels, teacher_channels, 1), nn.BatchNorm2d(teacher_channels))


    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        return {
            "backbone": backbone,
            "proposal_generator": build_proposal_generator(cfg, backbone.output_shape()),
            "roi_heads": build_roi_heads(cfg, backbone.output_shape()),
            "input_format": cfg.INPUT.FORMAT,
            "vis_period": cfg.VIS_PERIOD,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def visualize_training(self, batched_inputs, proposals):
        """
        A function used to visualize images and proposals. It shows ground truth
        bounding boxes on the original image and up to 20 top-scoring predicted
        object proposals on the original image. Users can implement different
        visualization functions for different models.

        Args:
            batched_inputs (list): a list that contains input to the model.
            proposals (list): a list that contains predicted proposals. Both
                batched_inputs and proposals should have the same length.
        """
        from detectron2.utils.visualizer import Visualizer

        storage = get_event_storage()
        max_vis_prop = 20

        for input, prop in zip(batched_inputs, proposals):
            img = input["image"]
            img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)
            v_gt = Visualizer(img, None)
            v_gt = v_gt.overlay_instances(boxes=input["instances"].gt_boxes)
            anno_img = v_gt.get_image()
            box_size = min(len(prop.proposal_boxes), max_vis_prop)
            v_pred = Visualizer(img, None)
            v_pred = v_pred.overlay_instances(
                boxes=prop.proposal_boxes[0:box_size].tensor.cpu().numpy()
            )
            prop_img = v_pred.get_image()
            vis_img = np.concatenate((anno_img, prop_img), axis=1)
            vis_img = vis_img.transpose(2, 0, 1)
            vis_name = "Left: GT bounding boxes;  Right: Predicted proposals"
            storage.put_image(vis_name, vis_img)
            break  # only visualize one image in a batch
    
    def image_vis(self, images):
        img = images[0].cpu().permute(1, 2, 0).numpy()
        cv2.imshow('img', img)
        cv2.waitKey(2500)
        pdb.set_trace()
    
    def KD_loss(self, student_logits, teacher_logits) :
        teacher_prob = F.softmax(teacher_logits, dim=1)
        student_log_prob = F.log_softmax(student_logits, dim=1)
        KD_loss = F.kl_div(student_log_prob, teacher_prob.detach(), reduction='batchmean')

        return KD_loss
    
    def NormalizeData(self, data):
        return (data - np.min(data)) / (np.max(data) - np.min(data))

    def forward(self, batched_inputs: List[Dict[str, torch.Tensor]], cfg=None, model_teacher=None, t_features=None, t_proposals=None, t_results=None, mode="test"):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances (optional): groundtruth :class:`Instances`
                * proposals (optional): :class:`Instances`, precomputed proposals.

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.

        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "instances" whose value is a :class:`Instances`.
                The :class:`Instances` object has the following keys:
                "pred_boxes", "pred_classes", "scores", "pred_masks", "pred_keypoints"
        """
        if not self.training and mode == "test":
            return self.inference(batched_inputs)
        images = self.preprocess_image(batched_inputs, mode)

        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None
 
        features = self.backbone(images.tensor)
        

        ddim_losses = {}
        t_s_losses = {}
        for i in ['p2', 'p3', 'p4', 'p5', 'p6']:
            # project student feature to the same dimension as teacher feature
            student_feat = features[i]
            teacher_feat = t_features[i]
            # student_feat = self.trans(student_feat)
            # use autoencoder on teacher feature
            if self.use_ae:
                hidden_t_feat, rec_t_feat = self.ae(teacher_feat)
                rec_loss = F.mse_loss(teacher_feat, rec_t_feat)
                teacher_feat = hidden_t_feat.detach()
            else:
                rec_loss = None
            refined_feat = self.pipeline(
                batch_size=student_feat.shape[0],
                device=student_feat.device,
                dtype=student_feat.dtype,
                shape=student_feat.shape[1:],
                feat=student_feat,
                num_inference_steps=self.diffusion_inference_steps,
                proj=self.proj
            )

            refined_feat = self.proj(refined_feat)
            ddim_losses[i+'_dimmloss'] = self.ddim_loss(teacher_feat)#计算扩散模型损失
            t_s_losses[i+'_ts_loss'] = self.t_s_loss(refined_feat,teacher_feat)
            features[i] = refined_feat+0.8*student_feat

        if self.proposal_generator is not None:
            
            proposals, proposal_losses = self.proposal_generator(images, features, t_results)
        else:
            assert "proposals" in batched_inputs[0]
            proposals = [x["proposals"].to(self.device) for x in batched_inputs]
            proposal_losses = {}
        
        results, detector_losses = self.roi_heads(images, features, proposals, t_results)

        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training(batched_inputs, proposals)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        
        losses.update(ddim_losses)
        losses.update(t_s_losses)

        
        return losses



    def ddim_loss(self, gt_feat):

        noise = torch.randn(gt_feat.shape, device=gt_feat.device) #.to(gt_feat.device)
        bs = gt_feat.shape[0]
        
        timesteps = torch.randint(0, self.scheduler.num_train_timesteps, (bs,), device=gt_feat.device).long()
        noisy_images = self.scheduler.add_noise(gt_feat, noise, timesteps)
        
        noise_pred = self.diffusionmodel(noisy_images, timesteps)
        loss = F.mse_loss(noise_pred, noise)
        return loss

    def inference(
        self,
        batched_inputs: List[Dict[str, torch.Tensor]],
        detected_instances: Optional[List[Instances]] = None,
        do_postprocess: bool = True,
    ):
        """
        Run inference on the given inputs.

        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`
            detected_instances (None or list[Instances]): if not None, it
                contains an `Instances` object per image. The `Instances`
                object contains "pred_boxes" and "pred_classes" which are
                known boxes in the image.
                The inference will then skip the detection of bounding boxes,
                and only predict other per-ROI outputs.
            do_postprocess (bool): whether to apply post-processing on the outputs.
            batched_inputs (list[dict]): 

        Returns:
            When do_postprocess=True, same as in :meth:`forward`.
            Otherwise, a list[Instances] containing raw network outputs.
        """
        assert not self.training

        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)

        
        for i in ['p2', 'p3', 'p4', 'p5', 'p6']:
            # project student feature to the same dimension as teacher feature
            student_feat = features[i]
            refined_feat = self.pipeline(
                batch_size=student_feat.shape[0],
                device=student_feat.device,
                dtype=student_feat.dtype,
                shape=student_feat.shape[1:],
                feat=student_feat,
                num_inference_steps=self.diffusion_inference_steps,
                proj=self.proj
            )
            features[i] = refined_feat+0.6*student_feat
        
        if detected_instances is None:
            if self.proposal_generator is not None:
                proposals, _ = self.proposal_generator(images, features, None)#生成提议
                # proposals, _ = self.proposal_generator(images, refined_feat, None)#生成提议
            else:
                assert "proposals" in batched_inputs[0]
                proposals = [x["proposals"].to(self.device) for x in batched_inputs]

            results, _ = self.roi_heads(images, features, proposals, None)
            # results, _ = self.roi_heads(images, refined_feat, proposals, None)
        else:
            detected_instances = [x.to(self.device) for x in detected_instances]
            results = self.roi_heads.forward_with_given_boxes(features, detected_instances)
            # results = self.roi_heads.forward_with_given_boxes(refined_feat, detected_instances)

        if do_postprocess:
            assert not torch.jit.is_scripting(), "Scripting is not supported for postprocess."
            return student_our_FPN_feature._postprocess(results, batched_inputs, images.image_sizes)
        else:
            return results
    
    def preprocess_image(self, batched_inputs: List[Dict[str, torch.Tensor]], mode = "test"):
        """
        Normalize, pad and batch the input images.
        """
        if mode == "train":
            images = [x["image_strong"].to(self.device) for x in batched_inputs]
            #self.image_vis(images)
            images = [(x - self.pixel_mean) / self.pixel_std for x in images]
            images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        elif mode == "test":#[{'file_name': 'dataset/VOC2012/JPEGImages/Metaplastic_188.jpg', 'image_id': 'Metaplastic_188', 'height': 1536, 'width': 2048, 'image': tensor
            images = [x["image"].to(self.device) for x in batched_inputs]#把图片放到gpu上
            images = [(x - self.pixel_mean) / self.pixel_std for x in images]#对每个图像张量 x 执行标准化操作
            images = ImageList.from_tensors(images, self.backbone.size_divisibility)#用于将一组图像张量转换为 ImageList 对象的操作
        return images

    @staticmethod
    def _postprocess(instances, batched_inputs: List[Dict[str, torch.Tensor]], image_sizes):
        """
        Rescale the output instances to the target size.
        """
        # note: private function; subject to changes
        processed_results = []

        for results_per_image, input_per_image, image_size in zip(
            instances, batched_inputs, image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append({"instances": r})
        return processed_results
    


    
