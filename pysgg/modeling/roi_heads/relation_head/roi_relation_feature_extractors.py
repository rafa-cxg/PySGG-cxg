# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import random

import ipdb
import numpy
import torch
from torch import nn

import cv2
from pysgg.modeling import registry
from pysgg.modeling.make_layers import make_fc
from pysgg.modeling.roi_heads.attribute_head.roi_attribute_feature_extractors import \
    make_roi_attribute_feature_extractor
from pysgg.modeling.roi_heads.box_head.roi_box_feature_extractors import make_roi_box_feature_extractor
from pysgg.structures.boxlist_ops import boxlist_union
from pysgg.modeling.roi_heads.relation_head.classifier import build_classifier
from pysgg.layers import (
    BatchNorm2d,
    Conv2d,
    FrozenBatchNorm2d,
    interpolate,
)
import matplotlib.pyplot as plt
from pysgg.data import get_dataset_statistics
from .utils_motifs import obj_edge_vectors
from pysgg.data.datasets.visual_genome import load_info
from pysgg.utils.imports import import_file
from pysgg.modeling.roi_heads.relation_head.patch_attention import ViT
from pysgg.structures.bounding_box import BoxList
ind_to_classes=['__background__', 'airplane', 'animal', 'arm', 'bag', 'banana', 'basket', 'beach', 'bear', 'bed', 'bench', 'bike', 'bird', 'board', 'boat', 'book', 'boot', 'bottle', 'bowl', 'box', 'boy', 'branch', 'building', 'bus', 'cabinet', 'cap', 'car', 'cat', 'chair', 'child', 'clock', 'coat', 'counter', 'cow', 'cup', 'curtain', 'desk', 'dog', 'door', 'drawer', 'ear', 'elephant', 'engine', 'eye', 'face', 'fence', 'finger', 'flag', 'flower', 'food', 'fork', 'fruit', 'giraffe', 'girl', 'glass', 'glove', 'guy', 'hair', 'hand', 'handle', 'hat', 'head', 'helmet', 'hill', 'horse', 'house', 'jacket', 'jean', 'kid', 'kite', 'lady', 'lamp', 'laptop', 'leaf', 'leg', 'letter', 'light', 'logo', 'man', 'men', 'motorcycle', 'mountain', 'mouth', 'neck', 'nose', 'number', 'orange', 'pant', 'paper', 'paw', 'people', 'person', 'phone', 'pillow', 'pizza', 'plane', 'plant', 'plate', 'player', 'pole','post', 'pot', 'racket', 'railing', 'rock', 'roof', 'room', 'screen', 'seat', 'sheep', 'shelf', 'shirt', 'shoe', 'short', 'sidewalk', 'sign', 'sink', 'skateboard', 'ski', 'skier', 'sneaker', 'snow', 'sock', 'stand', 'street', 'surfboard', 'table', 'tail', 'tie', 'tile', 'tire', 'toilet', 'towel', 'tower', 'track', 'train', 'tree', 'truck', 'trunk', 'umbrella', 'vase', 'vegetable', 'vehicle', 'wave', 'wheel', 'window', 'windshield', 'wing', 'wire', 'woman', 'zebra']
ind_to_predicates=['__background__', 'above', 'across', 'against', 'along', 'and', 'at', 'attached to', 'behind', 'belonging to', 'between', 'carrying', 'covered in', 'covering', 'eating', 'flying in', 'for', 'from', 'growing on', 'hanging from', 'has', 'holding', 'in', 'in front of', 'laying on', 'looking at', 'lying on', 'made of', 'mounted on', 'near', 'of', 'on', 'on back of', 'over', 'painted on', 'parked on', 'part of', 'playing', 'riding', 'says', 'sitting on', 'standing on', 'to', 'under', 'using', 'walking in', 'walking on', 'watching', 'wearing', 'wears', 'with']

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
                                                                    for_relation=False)#in_channels:256,for_relation=False??????union feature?????????????????????
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
            if self.separate_spatial:#todo ?????????????????????
                input_size = self.feature_extractor.resize_channels
                out_dim = self.feature_extractor.out_channels
                self.spatial_fc = nn.Sequential(*[make_fc(input_size, out_dim // 2), nn.ReLU(inplace=True),
                                                  make_fc(
                                                      out_dim // 2, out_dim), nn.ReLU(inplace=True),
                                                  ])
        self.visual_language_merger_edge = make_visual_language_merger_edge(
            cfg) if cfg.MODEL.ROI_RELATION_HEAD.VISUAL_LANGUAGE_MERGER_EDGE else None  # language???visual??????
    def forward(self, x, proposals, targets,rel_pair_idxs=None,visualize_feature=True):

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
            try:
                assert torch.max(rel_pair_idx[:, 1]).item()<80
            except :
                print(torch.max(rel_pair_idx[:, 0]).item())
            tail_proposal = proposal[rel_pair_idx[:, 1]]

            union_proposal = boxlist_union(head_proposal, tail_proposal)
            union_proposals.append(union_proposal)
            #??????sub obj ?????????union box?????????
            h=head_proposal.bbox; t=tail_proposal.bbox
            h1 = h[:,:2]-union_proposal.bbox[:,:2]
            h2 = h[:,2:]-union_proposal.bbox[:,:2]
            t1 = t[:, :2] - union_proposal.bbox[:, :2]
            t2 = t[:, 2:] - union_proposal.bbox[:, :2]
            union_proposal.add_field('relative_boxes',torch.cat((h1, h2, t1, t2), -1),is_custom=True)#??????head tail??????union box?????????
            # use range to construct rectangle, sized (rect_size, rect_size)
            num_rel = len(rel_pair_idx)
            dummy_x_range = torch.arange(self.rect_size, device=device).view(1, 1, -1).expand(num_rel, self.rect_size,
                                                                                              self.rect_size)
            dummy_y_range = torch.arange(self.rect_size, device=device).view(1, -1, 1).expand(num_rel, self.rect_size,
                                                                                              self.rect_size)
            # resize bbox to the scale rect_size
            if self.geometry_feature:#todo ??????????????????????????????????????????????????????????????????????????????box?????????4????????????range,????????????27*27????????????????????????????????????pooling,???7*7
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
        union_vis_features = self.feature_extractor.pooler(x, union_proposals)#?????????layer????????????pool,???????????????????????????
        if visualize_feature ==True:
            img_path = targets[0].get_field('image_paths')
            obj = targets[0].get_field('labels')
            coordinates=[]
            coordinates=[(u.bbox) for u in union_proposals]
            coordinates=torch.tensor(coordinates[0].detach())
            gap = nn.AdaptiveAvgPool2d(1)
            gt=[]
            rel_labels_all =targets[0].get_field('relation').long().detach().cpu()
            for pair_idx in rel_pair_idx:
                gt.append((ind_to_classes[int(obj[int(pair_idx[0])])],ind_to_predicates[int(int(rel_labels_all[pair_idx[0],pair_idx[1]]))],ind_to_classes[int(obj[int(pair_idx[1])])]))
            # gt=torch.tensor(gt)
            size=targets[0].size
            img = cv2.imread(img_path[0])  # ???cv2??????????????????
            size = get_size(img.shape[:2])
            img = cv2.resize(img, (size[1], size[0]))
            processed_ori = []
            for union_vis_feature,coordinate in zip(union_vis_features,coordinates):
                coordinate=coordinate.cpu().numpy()
                coordinate = coordinate.astype("int32")

                feature_map = union_vis_feature.squeeze(0).detach()
                weight=gap(feature_map)
                gray_scale=weight*feature_map
                gray_scale=torch.sum(gray_scale,dim=0).cpu().numpy()

                gray_scale = (gray_scale - numpy.min(gray_scale)) / (numpy.max(gray_scale) - numpy.min(gray_scale))
                # gray_scale = (gray_scale / numpy.sum(gray_scale))
                union = img[coordinate[1]:coordinate[3], coordinate[0]:coordinate[2], :]
                # try :
                #     assert(img.shape[1]>0 and img.shape[0]>0)
                # except:a=1
                heatmap = cv2.resize(gray_scale, (union.shape[1], union.shape[0]))  # ???????????????????????????????????????????????????
                heatmap = numpy.uint8(255 * heatmap)  # ?????????????????????RGB??????
                heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  # ?????????????????????????????????
                superimposed_img = cv2.addWeighted(union, 0.6, heatmap, 0.4, 0)
                superimposed_img = cv2.cvtColor(superimposed_img.astype(numpy.uint8), cv2.COLOR_BGR2RGB)
                # gray_scale = torch.sum(feature_map, 0)
                # gray_scale = gray_scale / feature_map.shape[0]
                # gray_scale = gray_scale / torch.sum(gray_scale)*255
                # torch.nn.functional.interpolate(gray_scale,size=size,mode='bilinear')
                processed_ori.append(superimposed_img)
            fig = plt.figure(figsize=(30, 50))
            count=0
            for i in range(len(processed_ori)):
                if gt[i][1]!='__background__' and count<20 and (gt[i][1]=='walking on' or gt[i][1]=='painteded on' or gt[i][1]=='mounted on' or gt[i][1]=='looking at'):
                    a = fig.add_subplot(5, 4, count + 1)
                    imgplot = plt.imshow(processed_ori[i])
                    a.axis("off")
                    a.set_title(gt[i], fontsize=30)
                    plt.savefig(str('feature_maps.jpg'), bbox_inches='tight')
                    count=count+1
        #????????????pair???language feature
        if self.cfg.MODEL.ROI_RELATION_HEAD.VISUAL_LANGUAGE_MERGER_EDGE:
            union_vis_features=self.visual_language_merger_edge(union_vis_features, proposals, rel_pair_idxs,union_proposals)
        if visualize_feature ==True:

            processed = []
            for union_vis_feature,coordinate in zip(union_vis_features,coordinates):
                coordinate = coordinate.cpu().numpy()
                coordinate = coordinate.astype("int32")
                feature_map = union_vis_feature.squeeze(0).detach()
                weight = gap(feature_map)
                gray_scale = weight * feature_map
                gray_scale = torch.sum(gray_scale, dim=0).cpu().numpy()

                gray_scale=(gray_scale-numpy.min(gray_scale))/(numpy.max(gray_scale)-numpy.min(gray_scale))
                # gray_scale = (gray_scale / numpy.sum(gray_scale))
                union = img[coordinate[1]:coordinate[3], coordinate[0]:coordinate[2], :]
                heatmap = cv2.resize(gray_scale, (union.shape[1], union.shape[0]))  # ???????????????????????????????????????????????????
                heatmap = numpy.uint8(255 * heatmap)  # ?????????????????????RGB??????
                heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  # ?????????????????????????????????

                superimposed_img = cv2.addWeighted(union, 0.6, heatmap, 0.4, 0)
                superimposed_img = cv2.cvtColor(superimposed_img.astype(numpy.uint8), cv2.COLOR_BGR2RGB)
                # cv2.imwrite(save_path, superimposed_img)  # ????????????????????????
                # gray_scale = torch.sum(feature_map, 0)
                # gray_scale = gray_scale / feature_map.shape[0]
                # gray_scale = gray_scale / torch.sum(gray_scale)*255
                processed.append(superimposed_img)
            fig = plt.figure(figsize=(30, 50))
            count = 0
            for i in range(len(processed)):
                if gt[i][1]!='__background__' and count<20 and (gt[i][1]=='walking on' or gt[i][1]=='painteded on' or gt[i][1]=='mounted on' or gt[i][1]=='looking at'):
                    a = fig.add_subplot(5, 4, count + 1)
                    imgplot = plt.imshow(processed[i])
                    a.axis("off")
                    a.set_title(gt[i], fontsize=30)
                    plt.savefig(str('feature_maps_c_bias.jpg'), bbox_inches='tight')
                    count=count+1
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
        self.latest_fusion=False#??????????????????
        self.early_fusion = True  #

        paths_catalog = import_file(
            "pysgg.config.paths_catalog", cfg.PATHS_CATALOG, True
        )
        dataset_names = cfg.DATASETS.TRAIN
        DatasetCatalog = paths_catalog.DatasetCatalog
        for dataset_name in dataset_names:
            data = DatasetCatalog.get(dataset_name, cfg)
            dict_file = data['args']['dict_file']
        self.obj_classes, self.rel_classes, self.ind_to_attributes = load_info(
            dict_file)

        self.num_obj_classes = len(self.obj_classes)
        self.num_rel_classes = len(self.rel_classes)
        self.pos_embed = nn.Sequential(*[
            nn.Linear(9, 32), nn.BatchNorm1d(32, momentum=0.001),
            nn.Linear(32, 200), nn.ReLU(inplace=True),
        ])
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
        if self.cfg.MODEL.ROI_RELATION_HEAD.USE_SEM_WORD_EMBEDDING:
            self.subemembedding = build_classifier(512, embed_dim)
            self.objemembedding = build_classifier(512, embed_dim)
            with torch.no_grad():
                self.subemembedding.reset_parameters()
                self.objemembedding.reset_parameters()
        else:#??????word2voc
            self.sublanguageembedding = nn.Embedding(self.num_obj_classes, embed_dim)
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
        #used for 7*7
        # self.ops = nn.Sequential(
        #     torch.nn.AvgPool2d(29, padding=2),
        #     torch.nn.Conv2d(1, 256, 3, 1, 1, bias=False),
        #     BatchNorm2d(256),
        #     nn.ReLU(inplace=True),
        #     # torch.nn.AvgPool2d(290, padding=2),
        #     torch.nn.Conv2d(256, 256, 3, 1, 1, bias=False),
        #     BatchNorm2d(256),
        #     nn.ReLU(inplace=True)
        # )
        dim=cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS
        #1*1
        # self.ops = nn.Sequential(
        #     torch.nn.AvgPool2d(29, padding=2),
        #     torch.nn.Conv2d(1, (dim//2), 3, 1, 1, bias=False),
        #     BatchNorm2d(dim//2),
        #     nn.ReLU(inplace=True),
        #     # torch.nn.AvgPool2d(290, padding=2),
        #     torch.nn.Conv2d(dim//2, dim, 3, 1, 1, bias=False),
        #     BatchNorm2d(dim),
        #     nn.ReLU(inplace=True),
        #     torch.nn.AvgPool2d(7)
        # )
        self.ops = nn.Sequential(*[
            torch.nn.AvgPool2d(29, padding=2),
            torch.nn.Conv2d(1, 256, 3, 1, bias=True),
            BatchNorm2d(256),
            nn.ReLU(inplace=True),
            torch.nn.Conv2d(256, 256, 3, 3, bias=True),
            BatchNorm2d(256),
            nn.ReLU(inplace=True)

        ])
        if self.cfg.MODEL.ROI_RELATION_HEAD.LANGUAGE_SPATIAL_ATTENTION:
            # #for 7*7map
            # self.ops2 = nn.Sequential(*[
            #     torch.nn.AvgPool2d(29, padding=2),
            #     torch.nn.Conv2d(1, 256, 3, 1,1, bias=False),
            #     BatchNorm2d(256),
            #     nn.ReLU(inplace=True),
            #     torch.nn.Conv2d(256, 256, 3, 1,1, bias=False),
            #     BatchNorm2d(256),
            #     nn.ReLU(inplace=True)
            # ])
            # # for calculate attention
            # self.ops3 = nn.Sequential(*[
            #
            #     torch.nn.Conv2d(256, 50, kernel_size=3, stride=1,padding=1),
            #     BatchNorm2d(50),
            #     nn.ReLU(inplace=True),
            #     torch.nn.Conv2d(50,256, kernel_size=3, stride=1,padding=1),
            #     nn.Sigmoid(),
            # ])
            self.vit = ViT(image_size=7,patch_size=1,num_classes=self.num_rel_classes,dim=256,depth = 2,heads = 4,dim_head=64,channels=256,mlp_dim = 512,dropout = 0.1,emb_dropout = 0.1)

        # self.motif_transform=make_fc(7*7*256,1024)
        # self.c1=Conv2d(1, 256, 3, 1, 1, bias=False)
        # self.bn=BatchNorm2d(256)
        # self.relu=nn.ReLU(inplace=True)
        # self.c2=Conv2d(256, 256, 3, 1, 1, bias=False)
        # self.bn2=BatchNorm2d(256)
        # self.avgpool=torch.nn.AvgPool2d(28,padding=2)

    def forward(self, visual_feature,proposals,rel_pair_idxs,union_proposals):
        # if self.latest_fusion:
            # visual=self.mlp1(input1)
            # language=self.mlp2(input2)
            # merge=visual+language
            # return merge
        # pos_embed = self.pos_embed(encode_box_info(proposals))
        if self.early_fusion:
            if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:  # obj_embed_by_pred_dist dim:200???obj_labels??????batch??????????????????
                obj_labels = torch.cat([proposal.get_field("labels") for proposal in proposals], dim=0).detach()


                subwordembedding_corpus = self.sublanguageembedding(
                    obj_labels.long())  # word embedding????????????word?????????embedding
                objwordembedding_corpus = self.objlanguageembedding(
                    obj_labels.long())

            else:
                if self.cfg.MODEL.ROI_RELATION_HEAD.use_possibility_merger:

                    obj_logits = torch.cat([proposal.get_field("predict_logits") for proposal in proposals], dim=0).detach()
                    subwordembedding_corpus = obj_logits @ self.sublanguageembedding.weight  # ?
                    objwordembedding_corpus = obj_logits @ self.objlanguageembedding.weight
                else:
                    obj_labels = torch.cat([proposal.get_field('pred_labels') for proposal in proposals], dim=0).detach()
                    subwordembedding_corpus = self.sublanguageembedding(
                    obj_labels.long())  # word embedding????????????word?????????embedding
                    objwordembedding_corpus = self.objlanguageembedding(
                    obj_labels.long())


            num = [len(proposal) for proposal in proposals]
            subwordembedding_corpus = subwordembedding_corpus.split(num, dim=0)
            objwordembedding_corpus = objwordembedding_corpus.split(num, dim=0)
            # subwordembedding_corpus = subwordembedding_corpus.split(num, dim=0)
            # objwordembedding_corpus = objwordembedding_corpus.split(num, dim=0)
            language_channel_attention=[]
            language_spatial_attention = []

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
                #?????????union box???sub obj?????????????????????merge????????????embedding???



                language_matrixs=((subwordembedding[rel_pair_idx[:,0]]).unsqueeze(-1) * (objwordembedding[rel_pair_idx[:,1]]).unsqueeze(-1).permute((0, 2, 1))).unsqueeze(1)
                op=self.ops(language_matrixs)

                language_channel_attention.append(op)
                # if self.cfg.MODEL.ROI_RELATION_HEAD.LANGUAGE_SPATIAL_ATTENTION:
                    # head_pos = self.pos_embed(encode_box_info(union_proposal.get_field("relative_boxes")[:,:4],union_proposal.size))
                    # tail_pos = self.pos_embed(encode_box_info(union_proposal.get_field("relative_boxes")[:, 4:],union_proposal.size))
                # union_proposal.del_field("relative_boxes")
                del language_matrixs; del op

            language_channel_attention=torch.cat(language_channel_attention,0)#[N_PAIRS,1,200,200]
            if self.cfg.MODEL.ROI_RELATION_HEAD.LANGUAGE_SPATIAL_ATTENTION:
                # language_spatial_attention = torch.cat(language_spatial_attention, 0)
                # visual_feature = visual_feature/ (visual_feature+1e-8).norm('fro',dim=(2,3), keepdim=True).cuda()
                # language_spatial_attention = language_spatial_attention/(language_spatial_attention+1e-8).norm('fro',dim=(2,3), keepdim=True).cuda()
                spatial_attention = self.vit(visual_feature+language_channel_attention)
                mixed=spatial_attention+(visual_feature+language_channel_attention)
            else:
                mixed = visual_feature + language_channel_attention
            return mixed
#???motif????????????????????????????????????
def encode_box_info(boxes,img_size):
    """
    encode proposed box information (x1, y1, x2, y2) to
    (cx/wid, cy/hei, w/wid, h/hei, x1/wid, y1/hei, x2/wid, y2/hei, wh/wid*hei)
    """



    wid = img_size[0]
    hei = img_size[1]
    wh = boxes[:, 2:] - boxes[:, :2] + 1.0
    xy = boxes[:, :2] + 0.5 * wh
    w, h = wh.split([1,1], dim=-1)
    x, y = xy.split([1,1], dim=-1)
    x1, y1, x2, y2 = boxes.split([1,1,1,1], dim=-1)
    assert wid * hei != 0
    info = torch.cat([w/wid, h/hei, x/wid, y/hei, x1/wid, y1/hei, x2/wid, y2/hei,
                      w*h/(wid*hei)], dim=-1).view(-1, 9)

    return info

def get_size(image_size):
    min_size=600
    max_size=1000
    if not isinstance(min_size, (list, tuple)):
        min_size = (min_size,)
    h,w = image_size
    size = random.choice(min_size)
    max_size = max_size
    if max_size is not None:
        min_original_size = float(min((w, h)))
        max_original_size = float(max((w, h)))
        if max_original_size / min_original_size * size > max_size:
            size = int(round(max_size * min_original_size / max_original_size))

    if (w <= h and w == size) or (h <= w and h == size):
        return (h, w)

    if w < h:
        ow = size
        oh = int(size * h / w)
    else:
        oh = size
        ow = int(size * w / h)

    return (oh, ow)