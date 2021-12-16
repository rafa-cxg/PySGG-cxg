# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import json

import torch

from pysgg.modeling.roi_heads.relation_head.rel_proposal_network.models import (
    gt_rel_proposal_matching,
    RelationProposalModel,
    filter_rel_pairs,
)
from pysgg.utils.visualize_graph import *
from .inference import make_roi_relation_post_processor
from .loss import make_roi_relation_loss_evaluator
from .roi_relation_feature_extractors import make_roi_relation_feature_extractor
from .roi_relation_predictors import make_roi_relation_predictor
from .sampling import make_roi_relation_samp_processor
from ..attribute_head.roi_attribute_feature_extractors import (
    make_roi_attribute_feature_extractor,
)
from ..box_head.roi_box_feature_extractors import (
    make_roi_box_feature_extractor,
    ResNet50Conv5ROIFeatureExtractor,
)
from pysgg.modeling.roi_heads.relation_head.model_kern import (
    to_onehot,
)
from ..two_stage_heads.two_stage_heads import make_Two_Stage_predictor

class ROIRelationHead(torch.nn.Module):
    """
    Generic Relation Head class.
    """

    def __init__(self, cfg, in_channels):
        super(ROIRelationHead, self).__init__()
        self.cfg = cfg.clone()
        self.usecluster = False
        self.num_obj_cls = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.num_rel_cls = cfg.MODEL.ROI_RELATION_HEAD.NUM_CLASSES

        # mode
        if cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
            if cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
                self.mode = "predcls"
            else:
                self.mode = "sgcls"
        else:
            self.mode = "sgdet"
        if cfg.USE_CLUSTER:
            self.usecluster=True#代表第一阶段标签是聚类后的
            with open('clustering/predicate2cluster.json', 'r') as f:
                self.predicate2cluster = json.load(f)
        # same structure with box head, but different parameters
        # these param will be trained in a slow learning rate, while the parameters of box head will be fixed
        # Note: there is another such extractor in uniton_feature_extractor
        self.union_feature_extractor = make_roi_relation_feature_extractor(
            cfg,
            in_channels,
        )
        if cfg.MODEL.ATTRIBUTE_ON:
            self.box_feature_extractor = make_roi_box_feature_extractor(
                cfg, in_channels, half_out=True
            )
            self.att_feature_extractor = make_roi_attribute_feature_extractor(
                cfg, in_channels, half_out=True
            )
            feat_dim = self.box_feature_extractor.out_channels * 2
        else:
            # the fix features head for extracting the instances ROI features for
            # obj detection
            self.box_feature_extractor = make_roi_box_feature_extractor(cfg, in_channels)
            feat_dim = self.box_feature_extractor.out_channels
            if isinstance(self.box_feature_extractor, ResNet50Conv5ROIFeatureExtractor):
                feat_dim = self.box_feature_extractor.flatten_out_channels

        self.predictor = make_roi_relation_predictor(cfg, feat_dim)
        self.post_processor = make_roi_relation_post_processor(cfg)
        self.loss_evaluator = make_roi_relation_loss_evaluator(cfg)
        self.samp_processor = make_roi_relation_samp_processor(cfg)

        self.rel_prop_on = self.cfg.MODEL.ROI_RELATION_HEAD.RELATION_PROPOSAL_MODEL.SET_ON
        self.rel_prop_type = self.cfg.MODEL.ROI_RELATION_HEAD.RELATION_PROPOSAL_MODEL.METHOD

        self.object_cls_refine = cfg.MODEL.ROI_RELATION_HEAD.OBJECT_CLASSIFICATION_REFINE
        self.pass_obj_recls_loss = cfg.MODEL.ROI_RELATION_HEAD.REL_OBJ_MULTI_TASK_LOSS

        # parameters
        self.use_union_box = self.cfg.MODEL.ROI_RELATION_HEAD.PREDICT_USE_VISION

        self.rel_pn_thres = torch.nn.Parameter(torch.Tensor([0.5]), requires_grad=False)
        self.rel_pn_thres_for_test = torch.nn.Parameter(
            torch.Tensor(
                [
                    0.33,
                ]
            ),
            requires_grad=False,
        )
        self.rel_pn = None
        self.use_relness_ranking = False
        self.use_same_label_with_clser = False
        if self.rel_prop_on:
            if self.rel_prop_type == "rel_pn":
                self.rel_pn = RelationProposalModel(cfg)
                self.use_relness_ranking = (
                    cfg.MODEL.ROI_RELATION_HEAD.RELATION_PROPOSAL_MODEL.USE_RELATEDNESS_FOR_PREDICTION_RANKING
                )
            if self.rel_prop_type == "pre_clser":
                self.use_same_label_with_clser == cfg.MODEL.ROI_RELATION_HEAD.RELATION_PROPOSAL_MODEL.USE_SAME_LABEL_WITH_CLSER

    def forward(self, features, proposal, targets=None, logger=None,**kwargs):
        """
        Arguments:
            features (list[Tensor]): feature-maps from possibly several levels
            proposals (list[BoxList]): proposal boxes. Note: it has been post-processed (regression, nms) in sgdet mode
            targets (list[BoxList], optional): the ground-truth targets.

        Returns:
            x (Tensor): the result of the feature extractor
            proposals (list[BoxList]): during training, the subsampled proposals
                are returned. During testing, the predicted boxlists are returned
            losses (dict[Tensor]): During training, returns the losses for the
                head. During testing, returns an empty dict.
        """

        if self.training:
            # relation subsamples and assign ground truth label during training
            with torch.no_grad():
                if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX: #precls
                    if kwargs=={}:#代表不用teostage
                        (#return的都是list:num_image
                            proposals,#[num_prop]
                            rel_labels,#[num_prop*(num_prop-1)]
                            rel_pair_idxs,#[num_prop*(num_prop-1),2]
                            gt_rel_binarys_matrix,#[num_prop,num_prop]
                        ) = self.samp_processor.gtbox_relsample(proposal, targets)#boxlist:num_image,对于predcls,proposals数目=targets
                        rel_labels_all = rel_labels
                    else:
                        if self.cfg.MODEL.TWO_STAGE_HEAD.UNION_BOX:
                            proposals,rel_labels,rel_pair_idxs,gt_rel_binarys_matrix,_,_=kwargs.values()
                        else:
                            proposals=kwargs['proposals']; rel_labels=kwargs['rel_labels']; rel_pair_idxs=kwargs['rel_pair_idxs']
                            gt_rel_binarys_matrix=kwargs['gt_rel_binarys_matrix']
                        rel_labels_all = rel_labels
                else:
                    if kwargs == {}:  # 代表不用teostage
                        (
                            proposals,
                            rel_labels,
                            rel_labels_all,
                            rel_pair_idxs,
                            gt_rel_binarys_matrix,
                        ) = self.samp_processor.detect_relsample(proposal, targets)
                    else:
                        if self.cfg.MODEL.TWO_STAGE_HEAD.UNION_BOX:
                            proposals, rel_labels,rel_labels_all, rel_pair_idxs, gt_rel_binarys_matrix,_,_ = kwargs.values()
                        else:
                            proposals, rel_labels, rel_pair_idxs, gt_rel_binarys_matrix = kwargs.values()
        else:
            rel_labels, rel_labels_all, gt_rel_binarys_matrix = None, None, None
            if kwargs == {}:  # 代表不用teostage
                rel_pair_idxs = self.samp_processor.prepare_test_pairs(#不超过设定的max relation数，就全部保留
                    features[0].device, proposal
                )
                proposals=proposal
            else:
                proposals=kwargs['proposals']
                rel_pair_idxs=kwargs['rel_pair_idxs']

        if self.mode == "predcls":
            # overload the pred logits by the gt label
            # gt_2stage=0
            if kwargs!= {}:
                rel_labels = kwargs['rel_labels']
                # rel_labels_all = kwargs['rel_labels_all']
                gt_rel_binarys_matrix = kwargs['gt_rel_binarys_matrix']
            device = features[0].device
            for proposal in proposals:
                obj_labels = proposal.get_field("labels")
                proposal.add_field("predict_logits", to_onehot(obj_labels, self.num_obj_cls))
                proposal.add_field("pred_scores", torch.ones(len(obj_labels)).to(device))#虽然key是pred..但对于predcls来说，都是确定的
                proposal.add_field("pred_labels", obj_labels.to(device))
                # proposal.add_field("gt_2stage", gt_2stage)

        # use box_head to extract features that will be fed to the later predictor processing
        roi_features = self.box_feature_extractor(features, proposals)#roi_align, roi_features:[num_prop,4096]
        if isinstance(self.box_feature_extractor, ResNet50Conv5ROIFeatureExtractor):
            roi_features = self.box_feature_extractor.flatten_roi_features(roi_features)

        rel_pn_loss = None
        relness_matrix = None
        if self.rel_prop_on:#true
            fg_pair_matrixs = None
            gt_rel_binarys_matrix = None

            if targets is not None:
                fg_pair_matrixs, gt_rel_binarys_matrix = gt_rel_proposal_matching(#fg_pair_matrixs指的是
                    proposals,
                    targets,
                    self.cfg.MODEL.ROI_HEADS.FG_IOU_THRESHOLD,#0.5
                    self.cfg.TEST.RELATION.REQUIRE_OVERLAP,#false
                )
                gt_rel_binarys_matrix = [each.float().cuda() for each in gt_rel_binarys_matrix]


            if self.rel_prop_type == "rel_pn":#rel proposal network . rel_prop_type=relawarefeature DEFAULT: FALSE#todo 确认rel_pn只能用在predcls中
                relness_matrix, rel_pn_loss = self.rel_pn(
                    proposals,
                    roi_features,
                    rel_pair_idxs,
                    rel_labels,
                    fg_pair_matrixs,
                    gt_rel_binarys_matrix,
                )

                rel_pair_idxs, rel_labels = filter_rel_pairs(
                    relness_matrix, rel_pair_idxs, rel_labels
                )
                for enti_prop, rel_mat in zip(proposals, relness_matrix):
                    enti_prop.add_field('relness_mat', rel_mat.unsqueeze(-1)) 

        if self.cfg.MODEL.ATTRIBUTE_ON:
            att_features = self.att_feature_extractor(features, proposals)
            roi_features = torch.cat((roi_features, att_features), dim=-1)

        if self.use_union_box:#yes
            union_features = self.union_feature_extractor(features, proposals, rel_pair_idxs)#union包括所有的N!组合
        else:
            union_features = None

        # final classifier that converts the features into predictions
        # should corresponding to all the functions and layers after the self.context class
        rel_pn_labels = rel_labels
        if not self.use_same_label_with_clser:
            rel_pn_labels = rel_labels_all


        obj_refine_logits, relation_logits, add_losses = self.predictor(#BGNNPredictor，
            proposals,#list
            rel_pair_idxs,#list
            rel_pn_labels,#list:根据rel_pair_idxs排
            gt_rel_binarys_matrix,
            roi_features,#tensor
            union_features,#tensor
            logger,
        )# 计算loss:RelAwareLoss. add_losses包括3个，分别对应3个iteration的predict预测loss
        relation_logits=list(relation_logits)
        if self.cfg.MODEL.TWO_STAGE_ON:
            # twostage_roi_features = twostage.box_feature_extractor(features, proposals)
            # twostage_union_feature=twostage.union_feature_extractor(features, proposals, rel_pair_idxs)
            # two_stage_logits = twostage.two_stage_predictor(  # twostagePredictor，
            #     proposals,
            #     rel_pair_idxs,
            #     rel_pn_labels,
            #     gt_rel_binarys_matrix,
            #     twostage_roi_features,
            #     twostage_union_feature,
            #     logger,
            # )


            # sactter_two_stage_logits_batch = []
            for idx, proposal in enumerate(proposals):
                sactter_two_stage_logits = torch.zeros(proposal.get_field('two_stage_pred_rel_logits').size(0),self.cfg.MODEL.ROI_RELATION_HEAD.NUM_CLASSES).type(torch.DoubleTensor).to('cuda')
                if self.usecluster:
                    for key, value in self.predicate2cluster.items():
                        sactter_two_stage_logits[:, [int(i) for i in value]] = proposal.get_field(
                            "two_stage_pred_rel_logits")[:, int(key) + 1].unsqueeze(-1).expand(-1, len(value))
                    sactter_two_stage_logits[:, 0] = proposal.get_field("two_stage_pred_rel_logits")[:, 0]
                    # sactter_two_stage_logits_batch.append(sactter_two_stage_logits)
                else: sactter_two_stage_logits=proposal.get_field("two_stage_pred_rel_logits")
                relation_logits[idx] = relation_logits[idx]*sactter_two_stage_logits#sactter_two_stage_logits



        # proposals, rel_pair_idxs, rel_pn_labels,relness_net_input,roi_features,union_features, None
        # for test
        if not self.training:
            # re-NMS on refined object prediction logits
            if not self.object_cls_refine:
                # if don't use object classification refine, we just use the initial logits
                obj_refine_logits = [prop.get_field("predict_logits") for prop in proposals]
            if kwargs!={}:
                two_stage_pred_rel_logits = [prop.get_field("two_stage_pred_rel_logits") for prop in proposals]
            else: two_stage_pred_rel_logits=[None for prop in proposals]
            '''post_processor只在test时期采用'''
            result = self.post_processor(#result包括extra_fields和box
                (relation_logits, obj_refine_logits,two_stage_pred_rel_logits), rel_pair_idxs, proposals
            )


            return roi_features, result, {}
        if self.cfg.MODEL.TRAIN_FIRST_STAGE_ONLY == False:#是否单训第一阶段
            loss_relation, loss_refine = self.loss_evaluator(#RelationLossComputation loss_refine指的是object的loss
                proposals, rel_labels, relation_logits, obj_refine_logits
            )

            output_losses = dict()
            if self.cfg.MODEL.ATTRIBUTE_ON and isinstance(loss_refine, (list, tuple)):
                output_losses = dict(
                    loss_rel=loss_relation,
                    loss_refine_obj=loss_refine[0],
                    loss_refine_att=loss_refine[1],
                )
            else:
                if self.pass_obj_recls_loss:#predcls:false output_losses:{}
                    output_losses = dict(loss_rel=loss_relation, loss_refine_obj=loss_refine)
                else:#default
                    output_losses = dict(loss_rel=loss_relation)

            if rel_pn_loss is not None:
                output_losses["loss_relatedness"] = rel_pn_loss

            output_losses.update(add_losses)#有rel的loss,也有4个iter的rel loss， 包括loss_rel和3个iter loss('pre_rel_classify_loss_iter-0'):属于rel_proposal_network部分
            output_losses_checked = {}#check whether loss is none
            if self.training:
                for key in output_losses.keys():
                    if output_losses[key] is not None:
                        if output_losses[key].grad_fn is not None:
                            output_losses_checked[key] = output_losses[key]
            output_losses = output_losses_checked

        else: output_losses=None
        return roi_features, proposals, output_losses#roi_features:[num_all_prop,4096]


def build_roi_relation_head(cfg, in_channels):
    """
    Constructs a new relation head.
    By default, uses ROIRelationHead, but if it turns out not to be enough, just register a new class
    and make it a parameter in the config
    """
    return ROIRelationHead(cfg, in_channels)
