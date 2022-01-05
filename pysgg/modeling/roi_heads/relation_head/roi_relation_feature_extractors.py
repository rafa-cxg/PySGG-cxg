# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import ipdb
import torch
from torch import nn

from pysgg.modeling import registry
from pysgg.modeling.make_layers import make_fc
from pysgg.modeling.roi_heads.attribute_head.roi_attribute_feature_extractors import \
    make_roi_attribute_feature_extractor
from pysgg.modeling.roi_heads.box_head.roi_box_feature_extractors import make_roi_box_feature_extractor
from pysgg.structures.boxlist_ops import boxlist_union

from pysgg.layers import (
    BatchNorm2d,
    Conv2d,
    FrozenBatchNorm2d,
    interpolate,
)
from pysgg.data import get_dataset_statistics
from .utils_motifs import obj_edge_vectors
@registry.ROI_RELATION_FEATURE_EXTRACTORS.register("RelationFeatureExtractor")
class RelationFeatureExtractor(nn.Module):
    """
    Heads for Motifs for relation triplet classification
    """

    def __init__(self, cfg, in_channels):
        super(RelationFeatureExtractor, self).__init__()
        self.cfg = cfg.clone()
        # should corresponding to obj_feature_map function in neural-motifs
        resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION#7
        pool_all_levels = cfg.MODEL.ROI_RELATION_HEAD.POOLING_ALL_LEVELS

        if cfg.MODEL.ATTRIBUTE_ON:
            self.feature_extractor = make_roi_box_feature_extractor(cfg, in_channels, half_out=True,
                                                                    cat_all_levels=pool_all_levels, for_relation=True)
            self.att_feature_extractor = make_roi_attribute_feature_extractor(cfg, in_channels, half_out=True,
                                                                              cat_all_levels=pool_all_levels)
            self.out_channels = self.feature_extractor.out_channels * 2
        else:
            self.feature_extractor = make_roi_box_feature_extractor(cfg, in_channels, cat_all_levels=pool_all_levels,
                                                                    for_relation=False)#in_channels:256,for_relation=False代表union feature提取不考虑全局
            self.out_channels = self.feature_extractor.out_channels#4096

        self.geometry_feature = cfg.MODEL.ROI_RELATION_HEAD.GEOMETRIC_FEATURES#true
        # union rectangle size
        self.rect_size = resolution * 4 - 1

        if self.geometry_feature:
            self.rect_conv = nn.Sequential(*[
                nn.Conv2d(2, in_channels // 2, kernel_size=7,
                          stride=2, padding=3, bias=True),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(in_channels // 2, momentum=0.01),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                nn.Conv2d(in_channels // 2, in_channels,
                          kernel_size=3, stride=1, padding=1, bias=True),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(in_channels, momentum=0.01),
            ])

            # separete spatial
            self.separate_spatial = self.cfg.MODEL.ROI_RELATION_HEAD.CAUSAL.SEPARATE_SPATIAL
            if self.separate_spatial:#todo 这是什么功能？
                input_size = self.feature_extractor.resize_channels
                out_dim = self.feature_extractor.out_channels
                self.spatial_fc = nn.Sequential(*[make_fc(input_size, out_dim // 2), nn.ReLU(inplace=True),
                                                  make_fc(
                                                      out_dim // 2, out_dim), nn.ReLU(inplace=True),
                                                  ])
        self.visual_language_merger_edge = make_visual_language_merger_edge(
            cfg) if cfg.MODEL.ROI_RELATION_HEAD.VISUAL_LANGUAGE_MERGER_EDGE else None  # language和visual融合
    def forward(self, x, proposals, rel_pair_idxs=None):
        device = x[0].device
        union_proposals = []
        rect_inputs = []

        for proposal, rel_pair_idx in zip(proposals, rel_pair_idxs):

            # try:
            #     assert (torch.max(rel_pair_idx[:, 0]).item() < 80)
            #     assert(torch.min(rel_pair_idx[:, 0]).item()>=0)
            #     assert(proposal.get_field("two_stage_pred_rel_logits").size()[0]>80)
            # except AssertionError:
            #     print(torch.max(rel_pair_idx[:, 0]).item())
            head_proposal = proposal[rel_pair_idx[:, 0]]
            assert torch.max(rel_pair_idx[:, 1]).item()<80
            tail_proposal = proposal[rel_pair_idx[:, 1]]

            union_proposal = boxlist_union(head_proposal, tail_proposal)

            union_proposals.append(union_proposal)

            # use range to construct rectangle, sized (rect_size, rect_size)
            num_rel = len(rel_pair_idx)
            dummy_x_range = torch.arange(self.rect_size, device=device).view(1, 1, -1).expand(num_rel, self.rect_size,
                                                                                              self.rect_size)
            dummy_y_range = torch.arange(self.rect_size, device=device).view(1, -1, 1).expand(num_rel, self.rect_size,
                                                                                              self.rect_size)
            # resize bbox to the scale rect_size
            if self.geometry_feature:#todo 我不理解这个特征的实际意义，目前我只是知道，原本一个box坐标就4个，通过range,变成一个27*27的二维信息，好进行卷积和pooling,到7*7
                head_proposal = head_proposal.resize(
                    (self.rect_size, self.rect_size))
                tail_proposal = tail_proposal.resize(
                    (self.rect_size, self.rect_size))
                head_rect = ((dummy_x_range >= head_proposal.bbox[:, 0].floor().view(-1, 1, 1).long())
                             & (dummy_x_range <= head_proposal.bbox[:, 2].ceil().view(-1, 1, 1).long())
                             & (dummy_y_range >= head_proposal.bbox[:, 1].floor().view(-1, 1, 1).long())
                             & (dummy_y_range <= head_proposal.bbox[:, 3].ceil().view(-1, 1, 1).long())).float()
                tail_rect = ((dummy_x_range >= tail_proposal.bbox[:, 0].floor().view(-1, 1, 1).long())
                             & (dummy_x_range <= tail_proposal.bbox[:, 2].ceil().view(-1, 1, 1).long())
                             & (dummy_y_range >= tail_proposal.bbox[:, 1].floor().view(-1, 1, 1).long())
                             & (dummy_y_range <= tail_proposal.bbox[:, 3].ceil().view(-1, 1, 1).long())).float()

                # (num_rel, 4, rect_size, rect_size)
                rect_input = torch.stack((head_rect, tail_rect), dim=1)#torch.Size([132, 2, 27, 27])
                rect_inputs.append(rect_input)

        # union visual feature. size (total_num_rel, in_channels, POOLER_RESOLUTION, POOLER_RESOLUTION)
        union_vis_features = self.feature_extractor.pooler(x, union_proposals)#对每一layer的特征算pool,在将通道拼接、缩减
        #这里加入pair的language feature
        if self.cfg.MODEL.ROI_RELATION_HEAD.VISUAL_LANGUAGE_MERGER_EDGE:
            union_vis_features=self.visual_language_merger_edge(union_vis_features, proposals, rel_pair_idxs)
        # merge two parts
        if self.geometry_feature:
            # rectangle feature. size (total_num_rel, in_channels, POOLER_RESOLUTION, POOLER_RESOLUTION)
            rect_inputs = torch.cat(rect_inputs, dim=0)
            rect_features = self.rect_conv(rect_inputs)

            if self.separate_spatial:
                region_features = self.feature_extractor.forward_without_pool(union_vis_features)
                spatial_features = self.spatial_fc(rect_features.view(rect_features.size(0), -1))
                union_features = (region_features, spatial_features)
            else:
                union_features = union_vis_features + rect_features
                union_features = self.feature_extractor.forward_without_pool(union_features)
                # (total_num_rel, out_channels)
        else:
            union_features = self.feature_extractor.forward_without_pool(union_vis_features)

        if self.cfg.MODEL.ATTRIBUTE_ON:
            union_att_features = self.att_feature_extractor.pooler(x, union_proposals)
            union_features_att = union_att_features + rect_features
            union_features_att = self.att_feature_extractor.forward_without_pool(union_features_att)
            union_features = torch.cat((union_features, union_features_att), dim=-1)

        return union_features


def make_roi_relation_feature_extractor(cfg, in_channels):
    func = registry.ROI_RELATION_FEATURE_EXTRACTORS[
        cfg.MODEL.ROI_RELATION_HEAD.FEATURE_EXTRACTOR
    ]
    return func(cfg, in_channels)#RelationFeatureExtractor
class make_visual_language_merger_edge(nn.Module):
    def __init__(self,cfg):
        super(make_visual_language_merger_edge, self).__init__()
        self.cfg=cfg
        # self.mlp1=nn.Sequential(
        #     make_fc(51, 51),
        #     nn.ReLU(True))
        # self.mlp2 = nn.Sequential(
        #     make_fc(51, 51),
        #     nn.ReLU(True))
        self.latest_fusion=False#在决策层相加
        self.early_fusion = True  #
        statistics = get_dataset_statistics(cfg)
        obj_classes, rel_classes = statistics['obj_classes'], statistics['rel_classes']
        self.num_obj_classes = len(obj_classes)
        self.num_rel_classes = len(rel_classes)
        self.obj_classes = obj_classes
        self.rel_classes = rel_classes
        # mode
        if cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
            if cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
                self.mode = "predcls"
            else:
                self.mode = "sgcls"
        else:
            self.mode = "sgdet"
        embed_dim = cfg.MODEL.ROI_RELATION_HEAD.EMBED_DIM
        obj_embed_vecs = obj_edge_vectors(self.obj_classes, wv_dir=cfg.GLOVE_DIR,
                                          wv_dim=embed_dim)  ##[151,200]
        self.sublanguageembedding = nn.Embedding(self.num_obj_classes,embed_dim)
        self.objlanguageembedding = nn.Embedding(self.num_obj_classes, embed_dim)

        with torch.no_grad():
            self.sublanguageembedding.weight.copy_(obj_embed_vecs, non_blocking=True)
            self.objlanguageembedding.weight.copy_(obj_embed_vecs, non_blocking=True)
        # self.ops =  nn.Sequential(*[
        #     torch.nn.AvgPool2d(8,padding=2),
        #     torch.nn.Conv2d(1, 51, 3, 1, bias=False),
        #     BatchNorm2d(51),
        #     nn.ReLU(inplace=True),
        #     torch.nn.Conv2d(51, 128, 1, 1, bias=False),
        #     BatchNorm2d(128),
        #     nn.ReLU(inplace=True),
        #     torch.nn.Conv2d(128, 256, 1, 1, bias=False),
        #     BatchNorm2d(256),
        #     torch.nn.AvgPool2d(3),
        # ])
        self.ops = nn.Sequential(*[
            torch.nn.AvgPool2d(29, padding=2),
            torch.nn.Conv2d(1, 256, 3, 1, bias=False),
            BatchNorm2d(256),
            nn.ReLU(inplace=True),
            torch.nn.Conv2d(256, 256, 3, 3, bias=False),
            BatchNorm2d(256),
            nn.ReLU(inplace=True)
        ])
        # self.c1=Conv2d(1, 256, 3, 1, 1, bias=False)
        # self.bn=BatchNorm2d(256)
        # self.relu=nn.ReLU(inplace=True)
        # self.c2=Conv2d(256, 256, 3, 1, 1, bias=False)
        # self.bn2=BatchNorm2d(256)
        # self.avgpool=torch.nn.AvgPool2d(28,padding=2)

    def forward(self, visual_feature,proposals,rel_pair_idxs):
        # if self.latest_fusion:
            # visual=self.mlp1(input1)
            # language=self.mlp2(input2)
            # merge=visual+language
            # return merge
        if self.early_fusion:
            if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:  # obj_embed_by_pred_dist dim:200。obj_labels是把batch里的拼在一起
                obj_labels = torch.cat([proposal.get_field("labels") for proposal in proposals], dim=0).detach()
                subwordembedding_corpus = self.sublanguageembedding(
                    obj_labels.long())  # word embedding层，输入word标签得embedding
                objwordembedding_corpus = self.objlanguageembedding(
                    obj_labels.long())
            else:
                # obj_logits = torch.cat([proposal.get_field("predict_logits") for proposal in proposals], dim=0).detach()
                # subwordembedding_corpus = obj_logits @ self.sublanguageembedding.weight  # ?
                # objwordembedding_corpus = obj_logits @ self.objlanguageembedding.weight
                obj_labels = torch.cat([proposal.get_field('pred_labels') for proposal in proposals], dim=0).detach()
                subwordembedding_corpus = self.sublanguageembedding(
                    obj_labels.long())  # word embedding层，输入word标签得embedding
                objwordembedding_corpus = self.objlanguageembedding(
                    obj_labels.long())

            num = [len(proposal) for proposal in proposals]
            subwordembedding_corpus = subwordembedding_corpus.split(num, dim=0)
            objwordembedding_corpus = objwordembedding_corpus.split(num, dim=0)
            # subwordembedding_corpus = subwordembedding_corpus.split(num, dim=0)
            # objwordembedding_corpus = objwordembedding_corpus.split(num, dim=0)
            languageembedding=[]

            # for subwordembedding,objwordembedding,rel_pair_idx in zip(subwordembedding_corpus,objwordembedding_corpus,rel_pair_idxs):
            #
            #     language_matrixs=(subwordembedding[rel_pair_idx.to('cpu')[:,0]].unsqueeze(-1) * objwordembedding[rel_pair_idx.to('cpu')[:,1]].unsqueeze(-1).permute((0, 2, 1))).unsqueeze(1).split(2,0)
            #     op=torch.cat([(self.ops(language_matrix.to('cuda'))).to('cpu') for language_matrix in language_matrixs],0)
            #     languageembedding.append(op)
            #     del language_matrixs; del op
            # languageembedding=torch.cat(languageembedding,0)#[N_PAIRS,1,200,200]
            # mixed=visual_feature+languageembedding.to('cuda')
            # return mixed
            for subwordembedding,objwordembedding,rel_pair_idx in zip(subwordembedding_corpus,objwordembedding_corpus,rel_pair_idxs):

                language_matrixs=(subwordembedding[rel_pair_idx[:,0]].unsqueeze(-1) * objwordembedding[rel_pair_idx[:,1]].unsqueeze(-1).permute((0, 2, 1))).unsqueeze(1)
                op=self.ops(language_matrixs)
                languageembedding.append(op)
                del language_matrixs; del op
            languageembedding=torch.cat(languageembedding,0)#[N_PAIRS,1,200,200]
            mixed=visual_feature+languageembedding
            return mixed

