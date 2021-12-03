# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch

from .attribute_head.attribute_head import build_roi_attribute_head
from .box_head.box_head import build_roi_box_head
from .keypoint_head.keypoint_head import build_roi_keypoint_head
from .mask_head.mask_head import build_roi_mask_head
from .relation_head.relation_head import build_roi_relation_head
from .two_stage_heads.two_stage_heads import build_two_stage_head


class CombinedROIHeads(torch.nn.ModuleDict):
    """
    Combines a set of individual heads (for box prediction or masks) into a single
    head.
    """

    def __init__(self, cfg, heads):
        super(CombinedROIHeads, self).__init__(heads)
        self.cfg = cfg.clone()
        if cfg.MODEL.MASK_ON and cfg.MODEL.ROI_MASK_HEAD.SHARE_BOX_FEATURE_EXTRACTOR:
            self.mask.feature_extractor = self.box.feature_extractor
        if cfg.MODEL.KEYPOINT_ON and cfg.MODEL.ROI_KEYPOINT_HEAD.SHARE_BOX_FEATURE_EXTRACTOR:
            self.keypoint.feature_extractor = self.box.feature_extractor

    def forward(self, features, proposals, targets=None, logger=None):
        losses = {}
        '''x:torch.Size([960, 4096])是nms筛选后的proposal feature.960=12*80
        如果是predcls:x:torch.Size([num_gt_allimage, 4096])  proposals:每个图片有1000+gt个 box '''
        x, detections, loss_box = self.box(features, proposals, targets)#ROIBoxHead的forward,x是proposal对应的visual feature.,detections是每个图的
        if not self.cfg.MODEL.RELATION_ON:
            # During the relationship training stage, the bbox_proposal_network should be fixed, and no loss. 
            losses.update(loss_box)

        if self.cfg.MODEL.TWO_STAGE_ON:
            detections,sampling,loss_two_stage = self.twostage(features, detections, targets, logger)#detection添加'two_stage_pred_rel_logits'
            losses.update(loss_two_stage)#此时Loss还没包含任何内容
        if self.cfg.MODEL.ATTRIBUTE_ON:
            # Attribute head don't have a separate feature extractor
            z, detections, loss_attribute = self.attribute(features, detections, targets)
            losses.update(loss_attribute)

        if self.cfg.MODEL.MASK_ON:
            mask_features = features
            # optimization: during training, if we share the feature extractor between
            # the box and the mask heads, then we can reuse the features already computed
            if (
                self.training
                and self.cfg.MODEL.ROI_MASK_HEAD.SHARE_BOX_FEATURE_EXTRACTOR
            ):
                mask_features = x
            # During training, self.box() will return the unaltered proposals as "detections"
            # this makes the API consistent during training and testing
            x, detections, loss_mask = self.mask(mask_features, detections, targets)
            losses.update(loss_mask)

        if self.cfg.MODEL.KEYPOINT_ON:
            keypoint_features = features
            # optimization: during training, if we share the feature extractor between
            # the box and the mask heads, then we can reuse the features already computed
            if (
                self.training
                and self.cfg.MODEL.ROI_KEYPOINT_HEAD.SHARE_BOX_FEATURE_EXTRACTOR
            ):
                keypoint_features = x
            # During training, self.box() will return the unaltered proposals as "detections"
            # this makes the API consistent during training and testing
            x, detections, loss_keypoint = self.keypoint(keypoint_features, detections, targets)
            losses.update(loss_keypoint)

        if self.cfg.MODEL.RELATION_ON:
            # it may be not safe to share features due to post processing
            # During training, self.box() will return the unaltered proposals as "detections"
            # this makes the API consistent during training and testing
            # with torch.no_grad():
            x, detections, loss_relation = self.relation(features, detections, targets, logger,sampling)#ROIRelationHead:在这里proposal被采样.x:roi_feature[all_prop,4096]. detections:proposal
            if loss_relation !=None:#不单独训第一阶段
                losses.update(loss_relation)#此时Loss还没包含任何内容

        return x, detections, losses


def build_roi_heads(cfg, in_channels):
    # individually create the heads, that will be combined together
    # afterwards
    roi_heads = []
    if cfg.MODEL.RETINANET_ON:
        return []

    if not cfg.MODEL.RPN_ONLY:
        roi_heads.append(("box", build_roi_box_head(cfg, in_channels)))
    if cfg.MODEL.MASK_ON:
        roi_heads.append(("mask", build_roi_mask_head(cfg, in_channels)))
    if cfg.MODEL.KEYPOINT_ON:
        roi_heads.append(("keypoint", build_roi_keypoint_head(cfg, in_channels)))
    if cfg.MODEL.RELATION_ON:
        roi_heads.append(("relation", build_roi_relation_head(cfg, in_channels)))
    if cfg.MODEL.ATTRIBUTE_ON:
        roi_heads.append(("attribute", build_roi_attribute_head(cfg, in_channels)))
    if cfg.MODEL.TWO_STAGE_ON:
        roi_heads.append(("twostage", build_two_stage_head(cfg, in_channels)))

    # combine individual heads in a single module
    if roi_heads:
         roi_heads = CombinedROIHeads(cfg, roi_heads)

    return roi_heads
