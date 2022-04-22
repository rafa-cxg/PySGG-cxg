# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Implements the Generalized R-CNN framework
"""

import torch
from torch import nn

from pysgg.structures.image_list import to_image_list

from ..backbone import build_backbone
from ..rpn.rpn import build_rpn
from ..roi_heads.roi_heads import build_roi_heads


class GeneralizedRCNN(nn.Module):
    """
    Main class for Generalized R-CNN. Currently supports boxes and masks.
    It consists of three main parts:
    - backbone
    - rpn
    - heads: takes the features + the proposals from the RPN and computes
        detections / masks from it.
    """

    def __init__(self, cfg):
        super(GeneralizedRCNN, self).__init__()
        self.cfg = cfg.clone()
        self.backbone = build_backbone(cfg)
        self.rpn = build_rpn(cfg, self.backbone.out_channels)#256
        self.roi_heads = build_roi_heads(cfg, self.backbone.out_channels)
        # self.two_stage_heads = build_roi_heads(cfg, self.backbone.out_channels)

    def forward(self, images, targets=None, logger=None):
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[BoxList]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        images  = to_image_list(images)
        features = self.backbone(images.tensors)#进入resnet.py，resnet class forward中，返回5-list是返回的resnet5个layer的结果.images.tensors:torch.Size([12, 3, 1024, 608]) 怎么做到尺寸一致的？
        proposals, proposal_losses = self.rpn(images, features, targets)#targets:box_list.proposals:仅有objectness
        if self.roi_heads:
            x, result, detector_losses = self.roi_heads(features, proposals, targets, logger)#CombinedROIHeads proposals:在100的基础上再加gt
        else:
            # RPN-only models don't have roi_heads
            x = features
            result = proposals
            detector_losses = {}

        if self.training:
            losses = {}
            losses.update(detector_losses)#'loss_rel'
            if not self.cfg.MODEL.RELATION_ON: #defalut is True
                # During the relationship training stage, the rpn_head should be fixed, and no loss. 
                losses.update(proposal_losses)#proposal_losses包括'loss_objectness' 'loss_rpn_box_reg'
            return losses

        return result
