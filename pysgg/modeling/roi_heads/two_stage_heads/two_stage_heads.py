# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import copy
import json
import pickle

import torch
from pysgg.data import get_dataset_statistics
from pysgg.modeling.roi_heads.relation_head.rel_proposal_network.models import (
    gt_rel_proposal_matching,
    RelationProposalModel,
    filter_rel_pairs,
)
from pysgg.utils.visualize_graph import *
from ..relation_head.inference import make_roi_relation_post_processor
from ..relation_head.loss import make_two_stage_loss_evaluator,make_loss_evaluator_distribution
from ..relation_head.roi_relation_feature_extractors import make_roi_relation_feature_extractor
from .two_stage_predictors import make_Two_Stage_predictor
from ..relation_head.sampling import make_two_stage_samp_processor
from ..relation_head.utils_motifs import encode_rel_box_info

from ..attribute_head.roi_attribute_feature_extractors import (
    make_roi_attribute_feature_extractor,
)
from ..box_head.roi_box_feature_extractors import (
    make_roi_box_feature_extractor,
    ResNet50Conv5ROIFeatureExtractor,
)
from ..two_stage_heads.two_stage_predictors import TwoStagePredictor
from pysgg.modeling.roi_heads.relation_head.model_kern import (
    to_onehot,
)

from torch import nn
from pysgg.modeling.make_layers import make_fc
from pysgg.modeling.utils import cat
from torch.nn import functional as F

from ..relation_head.utils_motifs import obj_edge_vectors, encode_box_info,env_pos_rel_box_info

class Two_Stage_Head(torch.nn.Module):
    """
    Generic Relation Head class.
    """

    def __init__(self, cfg, in_channels):
        super(Two_Stage_Head, self).__init__()
        self.cfg = cfg.clone()
        statistics = get_dataset_statistics(self.cfg)
        obj_classes, rel_classes = statistics['obj_classes'], statistics['rel_classes']
        self.num_obj_classes = len(obj_classes)
        self.num_rel_classes = len(rel_classes)
        self.obj_classes = obj_classes
        self.rel_classes = rel_classes
        self.data_dir = 'datasets/vg/'
        self.cluster_dir='clustering/'
        self.num_obj_cls = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES#151
        self.num_rel_cls = cfg.MODEL.TWO_STAGE_HEAD.NUM_REL_GROUP
        self.geometry_feat_dim = 128
        self.embed_dim = self.cfg.MODEL.ROI_RELATION_HEAD.EMBED_DIM
        self.pos_embed = nn.Sequential(*[
            make_fc(9, 32), nn.BatchNorm1d(32, momentum=0.001),
            make_fc(32, self.geometry_feat_dim), nn.ReLU(inplace=True),
        ])
        # self.pair_pos_embed = nn.Sequential(*[#描述中心点距离，overlap,相对位置
        #     make_fc(3, 32), nn.BatchNorm1d(32, momentum=0.001),
        #     make_fc(32, self.geometry_feat_dim), nn.ReLU(inplace=True),
        # ])
        obj_embed_vecs = obj_edge_vectors(self.obj_classes, wv_dir=self.cfg.GLOVE_DIR, wv_dim=self.embed_dim)##[151,200]
        self.obj_embed_on_prob_dist = nn.Embedding(self.num_obj_classes, self.embed_dim)
        self.obj_embed_on_pred_label = nn.Embedding(self.num_obj_classes, self.embed_dim)
        with torch.no_grad():
            self.obj_embed_on_prob_dist.weight.copy_(obj_embed_vecs, non_blocking=True)
            self.obj_embed_on_pred_label.weight.copy_(obj_embed_vecs, non_blocking=True)
        # self.num_rel_group =3
        if cfg.USE_CLUSTER:
            self.predicatename2cluster={}
            self.predicateid2cluster = {}
            with open(os.path.join(self.data_dir, 'predicate2cluster.json'), 'r') as json_file:
                predicatename2cluster = json.load(json_file)
                for key, value in predicatename2cluster.items():
                    for v in value:
                        self.predicatename2cluster[v] = key
            with open(os.path.join(self.cluster_dir, 'predicate2cluster.json'), 'r') as json_file:
                predicateid2cluster = json.load(json_file)
                for key, value in predicateid2cluster.items():
                    for v in value:
                        self.predicateid2cluster[int(v)] = int(key)+1#留出backgroud位置
            '''外加ground'''
            self.predicateid2cluster[0]=0

        self.loss_distribution=True if (cfg.MODEL.TWO_STAGE_HEAD.loss_distribution and  cfg.MODEL.TWO_STAGE_ON) else False
        # mode
        if cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
            if cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
                self.mode = "predcls"
            else:
                self.mode = "sgcls"
        else:
            self.mode = "sgdet"

        # same structure with box head, but different parameters
        # these param will be trained in a slow learning rate, while the parameters of box head will be fixed
        # Note: there is another such extractor in uniton_feature_extractor
        if cfg.MODEL.TWO_STAGE_HEAD.UNION_BOX:
            self.union_feature_extractor = make_roi_relation_feature_extractor(#RelationFeatureExtractor
                cfg,
                in_channels,#256
            )


        # the fix features head for extracting the instances ROI features for
        # obj detection
        if cfg.MODEL.TWO_STAGE_HEAD.INDIVIDUAL_BOX:
             self.box_feature_extractor = make_roi_box_feature_extractor(cfg, in_channels)#FPN2MLPFeatureExtractor，来自/box_head/
             feat_dim = self.box_feature_extractor.out_channels  # 4096
             if isinstance(self.box_feature_extractor, ResNet50Conv5ROIFeatureExtractor):
                 feat_dim = self.box_feature_extractor.flatten_out_channels
        if cfg.MODEL.TWO_STAGE_HEAD.PURE_SENMENTIC:
            self.pure_senmatic_feature=make_pure_senmatic_feature(cfg,in_channels)
            feat_dim=4096#权宜之计，这里并不用到它
        self.make_prior_distribution = make_prior_distribution(cfg, in_channels)

        if cfg.MODEL.TWO_STAGE_ON:
            self.two_stage_predictor = make_Two_Stage_predictor(cfg, feat_dim)

        self.post_processor = make_roi_relation_post_processor(cfg)
        self.loss_evaluator_2stage = make_two_stage_loss_evaluator(cfg)
        self.loss_evaluator_distribution = make_loss_evaluator_distribution(cfg)
        self.samp_processor = make_two_stage_samp_processor(cfg)

        self.rel_prop_on = self.cfg.MODEL.ROI_RELATION_HEAD.RELATION_PROPOSAL_MODEL.SET_ON
        self.rel_prop_type = self.cfg.MODEL.ROI_RELATION_HEAD.RELATION_PROPOSAL_MODEL.METHOD

        # parameters
        self.use_union_box = self.cfg.MODEL.TWO_STAGE_HEAD.UNION_BOX

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

    def forward(self, features, proposals, targets=None, logger=None):
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

        if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:#obj_embed_by_pred_dist dim:200。obj_labels是把batch里的拼在一起
            obj_labels = cat([proposal.get_field("labels") for proposal in proposals], dim=0).detach()
            obj_embed_by_pred_dist = self.obj_embed_on_prob_dist(obj_labels.long())#word embedding层，输入word标签得embedding
        else:
            obj_logits = cat([proposal.get_field("predict_logits") for proposal in proposals], dim=0).detach()
            obj_embed_by_pred_dist = F.softmax(obj_logits, dim=1) @ self.obj_embed_on_prob_dist.weight#?
        # box positive geometry embedding
        pos_embed = self.pos_embed(encode_box_info(proposals))  # 这个文章用的position embedding

        if self.training:
            # relation subsamples and assign ground truth label during training
            with torch.no_grad():
                if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX: #precls
                    (#return的都是list:num_image
                        proposals,#boxlist:num_img 与输入的oroo相比，怎加'locating_match'
                        rel_labels,#list:num_img
                        rel_pair_idxs,
                        gt_rel_binarys_matrix,
                    ) = self.samp_processor.gtbox_relsample(proposals, targets)#proposals:boxlist:num_image,对于predcls,proposals数目=targets
                    rel_labels_all=rel_labels
                    sampling = dict(proposals=proposals, rel_labels=rel_labels,
                                    rel_pair_idxs=rel_pair_idxs, gt_rel_binarys_matrix=gt_rel_binarys_matrix)
                else:
                    (
                        proposals,#[num_prop]
                        rel_labels,#[num_prop*(num_prop-1)]
                        rel_labels_all,#[num_prop*(num_prop-1),2]todo rel_all和rel_label区别？
                        rel_pair_idxs,#[num_prop*(num_prop-1),2]
                        gt_rel_binarys_matrix,#[num_prop,num_prop]
                    ) = self.samp_processor.detect_relsample(proposals, targets)
                    sampling=dict(proposals=(proposals),rel_labels=copy.deepcopy(rel_labels),rel_labels_all=rel_labels_all,rel_pair_idxs=(rel_pair_idxs),gt_rel_binarys_matrix=gt_rel_binarys_matrix)
                    # sampling={}
                if self.cfg.USE_CLUSTER==True:
                    rel_labels_2stage = []
                    for rel_label in rel_labels:
                        _ = [int(self.predicateid2cluster[int(f)]) for f in rel_label]
                        rel_labels_2stage.append(torch.LongTensor(_).to('cuda'))
                    rel_labels_all = rel_labels_2stage
        else:
            rel_labels, rel_labels_all, rel_labels_2stage,gt_rel_binarys_matrix = None, None, None,None

            rel_pair_idxs = self.samp_processor.prepare_test_pairs(#不超过设定的max relation数，就全部保留
                features[0].device, proposals
            )
            sampling = dict(proposals=proposals, rel_labels=rel_labels,rel_labels_all=rel_labels_all,
                            rel_pair_idxs=rel_pair_idxs, gt_rel_binarys_matrix=gt_rel_binarys_matrix)
        #todo sgdet的memory不够用，先注释
        # embed = env_pos_rel_box_info(proposals, rel_pair_idxs)#这是transformer位置编码
        # sampling.update(embed=embed)
        embed=None

        if self.mode \
                == "predcls":
            # overload the pred logits by the gt label
            device = features[0].device
        if self.cfg.MODEL.TWO_STAGE_HEAD.PURE_SENMENTIC==False:
            # use box_head to extract features that will be fed to the later predictor processing
            roi_features = self.box_feature_extractor(features, proposals)#roi_align, roi_features:[num_all_prop,4096]
            if isinstance(self.box_feature_extractor, ResNet50Conv5ROIFeatureExtractor):
                roi_features = self.box_feature_extractor.flatten_roi_features(roi_features)
            obj_rep = cat((roi_features, obj_embed_by_pred_dist, pos_embed), -1)  # 4096,200,128.->4424
        else:#纯语义特征
            roi_features=None

            obj_rep=self.pure_senmatic_feature(proposals, rel_pair_idxs,obj_embed_by_pred_dist)
            distribution=self.make_prior_distribution (proposals, rel_pair_idxs)


        if self.cfg.MODEL.TWO_STAGE_HEAD.INDIVIDUAL_BOX:
            sampling.update(obj_rep=obj_rep)
        if self.rel_prop_on:#true
            fg_pair_matrixs = None
            gt_rel_binarys_matrix = None

            if targets is not None:
                fg_pair_matrixs, gt_rel_binarys_matrix = gt_rel_proposal_matching(#prop中可以和gt对应上的matrix
                    proposals,#predcls:除对角线，全是1
                    targets,
                    self.cfg.MODEL.ROI_HEADS.FG_IOU_THRESHOLD,#0.5
                    self.cfg.TEST.RELATION.REQUIRE_OVERLAP,#false
                )
                gt_rel_binarys_matrix = [each.float().cuda() for each in gt_rel_binarys_matrix]

        if self.use_union_box:#yes
            union_features = self.union_feature_extractor(features, proposals, rel_pair_idxs)#[all_pair,4096]
            sampling.update(union_features=union_features)
        # '''先不写判断，直接加位置'''
        else:
            union_features = None



        # final classifier that converts the features into predictions
        # should corresponding to all the functions and layers after the self.context class
        rel_pn_labels = rel_labels
        if not self.use_same_label_with_clser:
            rel_pn_labels = rel_labels_all

        if isinstance(self.two_stage_predictor, TwoStagePredictor):
            relation_logits = self.two_stage_predictor(
                proposals,
                rel_pair_idxs,
                rel_pn_labels,
                gt_rel_binarys_matrix,
                roi_features,
                union_features,
                obj_rep,
                embed,
                logger)# 计算loss:RelAwareLoss. add_losses包括4个，分别对应4个iteration的predict预测loss RelAwareLoss
        else:
            sampling.update(rel_labels_2stage=rel_labels_2stage)#?
            relation_logits = self.two_stage_predictor(features[3],**sampling)#[N_pair,num_group]
         # proposals, rel_pair_idxs, rel_pn_labels,relness_net_input,roi_features,union_features, None
        # for test
        for proposal, two_stage_logit,dist in zip(proposals, relation_logits,distribution):
            # two_stage_logit = F.softmax(two_stage_logit, -1)  # 传给第二阶段的logit限制在0-1
            proposal.del_field('center')
            proposal.add_field("two_stage_pred_rel_logits", two_stage_logit,is_custom=True)#在cpu上运行

        if not self.training:
            result =proposals


            return  result,sampling, {}
        if self.cfg.MODEL.TWO_STAGE_HEAD.USE_TWOSTAGE_LOSS==False:
            output_losses=None
            pass
        else:
            if self.loss_distribution:
                try:
                    distribution
                except NameError:
                    print("Check in whether \"PURE_SENMENTIC\" is ON in cfg file!")


                loss_relation =self.loss_evaluator_distribution(relation_logits,rel_labels_all,distribution)
            else:
                loss_relation = self.loss_evaluator_2stage(
                proposals, rel_labels_all, relation_logits
            )
            #torch.argmax(relation_logits[0][:,1:],-1)+1
            output_losses = dict()

            output_losses = dict(loss_two_stage=loss_relation)
            output_losses_checked = {}#check whether loss is none
            if self.training:
                for key in output_losses.keys():
                    if output_losses[key] is not None:
                        if output_losses[key].grad_fn is not None:
                            output_losses_checked[key] = output_losses[key]
            output_losses = output_losses_checked


        return  proposals,sampling, output_losses#roi_features:[num_all_prop,4096]


def build_two_stage_head(cfg, in_channels):
    """
    Constructs a new relation head.
    By default, uses ROIRelationHead, but if it turns out not to be enough, just register a new class
    and make it a parameter in the config
    """
    return Two_Stage_Head(cfg, in_channels)
class  make_pure_senmatic_feature(torch.nn.Module):
    def __init__(self, cfg, in_channels):
        super(make_pure_senmatic_feature, self).__init__()
        self.cfg=cfg
        self.type= 'glove' if self.cfg.MODEL.TWO_STAGE_HEAD.USE_GLOVE else 'distribution'#选择采用glove还是先验作为word embedding
        if self.type=='distribution':
            with open(os.path.join(self.cfg.OUTPUT_DIR, "record_sub_distribution.pkl"), 'rb') as f1:
                self.sub_distribution = torch.tensor(pickle.load(f1).astype(float)).clone().detach()
            with open(os.path.join(self.cfg.OUTPUT_DIR, "record_obj_distribution.pkl"), 'rb') as f2:
                self.obj_distribution = torch.tensor(pickle.load(f2).astype(float)).clone().detach()
        self.in_channels=in_channels

        self.sub_embed = nn.Sequential(
            *[
                make_fc(200, 1024),
                nn.ReLU(inplace=True),
            ]
        )
        self.obj_embed=nn.Sequential(
            *[
                make_fc(200,1024),
                nn.ReLU(inplace=True),
            ]
        )
        self.interact_embed = nn.Sequential(
            *[
                make_fc(9*2+2, 1024),
                nn.ReLU(inplace=True),
            ]
        )
        # mode
        if cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
            if cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
                self.mode = "predcls"
            else:
                self.mode = "sgcls"
        else:
            self.mode = "sgdet"

    def forward(self,proposals,rel_pair_idxs,wordembedding_corpus):
        prop_infos=encode_box_info(proposals)#x,y...


        rel_infos=(encode_rel_box_info(proposals, rel_pair_idxs))#iou,distance
        sub_embed=[]
        obj_embed=[]
        interact_embeds=[]
        pos_embed=[]
        # for image,rel_pair_idx in enumerate(rel_pair_idxs):
        #     sub_embed.append(self.sub_distribution[proposals[image].get_field('labels').type(torch.LongTensor)[rel_pair_idx[:, 0]]])
        #     obj_embed.append(self.obj_distribution[proposals[image].get_field('labels').type(torch.LongTensor)[rel_pair_idx[:, 1]]])
        # sub_embed =torch.cat(sub_embed).cuda()
        # obj_embed =torch.cat(obj_embed).cuda()
        if self.type=='distribution':
            for image,(rel_pair_idx,rel_info) in enumerate(zip(rel_pair_idxs,rel_infos)):
                interact_embed = self.interact_embed(rel_info)
                interact_embeds.append(interact_embed)
                if self.mode=='predcls':
                    sub_embed.append(self.sub_distribution[proposals[image].get_field('labels').type(torch.LongTensor)[rel_pair_idx[:, 0]]])
                    obj_embed.append(self.obj_distribution[proposals[image].get_field('labels').type(torch.LongTensor)[rel_pair_idx[:, 1]]])

                else:
                    # label=torch.argmax(proposals[image].get_field('predict_logits'),-1)
                    sub_embed.append(self.sub_distribution[proposals[image].get_field('pred_labels').type(torch.LongTensor)[rel_pair_idx[:, 0]]])
                    obj_embed.append(self.obj_distribution[proposals[image].get_field('pred_labels').type(torch.LongTensor)[rel_pair_idx[:, 1]]])
            sub_embed =self.sub_embed(torch.cat(sub_embed).cuda().type(torch.cuda.FloatTensor))
            obj_embed =self.obj_embed(torch.cat(obj_embed).cuda().type(torch.cuda.FloatTensor))
        else:
            num=[len(proposal) for proposal in proposals]
            wordembedding_corpus = wordembedding_corpus.split(num, dim=0)
            prop_infos =  prop_infos.split(num, dim=0)
            for image, (rel_pair_idx,wordembedding,prop_info,rel_info) in enumerate(zip(rel_pair_idxs,wordembedding_corpus,prop_infos,rel_infos)):

                sub_embed.append(wordembedding[rel_pair_idx[:, 0]])
                obj_embed.append(wordembedding[rel_pair_idx[:, 1]])
                pos_embed.append(torch.cat((prop_info[rel_pair_idx[:, 0]],prop_info[rel_pair_idx[:, 1]],rel_info),1))


            sub_embed = self.sub_embed(torch.cat(sub_embed).cuda().type(torch.cuda.FloatTensor))
            obj_embed = self.obj_embed(torch.cat(obj_embed).cuda().type(torch.cuda.FloatTensor))
            pos_embed= self.interact_embed(torch.cat(pos_embed,0))
        return  torch.cat((sub_embed,obj_embed,pos_embed),-1)


class  make_prior_distribution(torch.nn.Module):
    def __init__(self, cfg, in_channels):
        super(make_prior_distribution, self).__init__()
        self.cfg=cfg
        self.softmax = nn.Softmax(dim=1)
        with open(os.path.join(self.cfg.OUTPUT_DIR, "record_sub_distribution.pkl"), 'rb') as f1:
            self.sub_distribution = torch.tensor(pickle.load(f1)).clone().detach().to('cuda')
        with open(os.path.join(self.cfg.OUTPUT_DIR, "record_obj_distribution.pkl"), 'rb') as f2:
            self.obj_distribution = torch.tensor(pickle.load(f2)).clone().detach().to('cuda')
        # mode
        if cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
            if cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
                self.mode = "predcls"
            else:
                self.mode = "sgcls"
        else:
            self.mode = "sgdet"
    def forward(self,proposals,rel_pair_idxs):
        embed=[]
        with torch.no_grad():
            for image, rel_pair_idx in enumerate(rel_pair_idxs):
                if self.mode == 'predcls':
                    prior=((self.sub_distribution[
                        proposals[image].get_field('labels').type(torch.LongTensor)[rel_pair_idx[:, 0]]]) * (
                     self.obj_distribution[
                         proposals[image].get_field('labels').type(torch.LongTensor)[rel_pair_idx[:, 1]]]))
                    prior=torch.nn.functional.normalize(prior,p=1)
                    embed.append(prior)

                else:
                    # label=torch.argmax(proposals[image].get_field('predict_logits'),-1)
                    embed.append(self.sub_distribution[proposals[image].get_field('pred_labels').type(torch.LongTensor)[
                        rel_pair_idx[:, 0]]]*self.obj_distribution[proposals[image].get_field('pred_labels').type(torch.LongTensor)[
                        rel_pair_idx[:, 1]]])
        return embed
        # return torch.cat(embed).cuda().type(torch.cuda.FloatTensor)
