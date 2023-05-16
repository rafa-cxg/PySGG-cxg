# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch import nn
from torch.nn import functional as F

from pysgg.modeling import registry
from pysgg.modeling.backbone import resnet
from pysgg.modeling.make_layers import group_norm
from pysgg.modeling.make_layers import make_fc
from pysgg.modeling.poolers import Pooler
from pysgg.data.datasets.visual_genome import load_info
from pysgg.utils.imports import import_file
from ..relation_head.utils_motifs import obj_edge_vectors
from pysgg.layers import (
    BatchNorm2d,
    Conv2d,
    FrozenBatchNorm2d,
    interpolate,
)
from ..relation_head.model_transformer import TransformerEncoder
from ..relation_head.utils_motifs import obj_edge_vectors, to_onehot, encode_box_info
from pysgg.data import get_dataset_statistics
@registry.ROI_BOX_FEATURE_EXTRACTORS.register("ResNet50Conv5ROIFeatureExtractor")
class ResNet50Conv5ROIFeatureExtractor(nn.Module):
    def __init__(self, cfg, in_channels, half_out=False, cat_all_levels=False, for_relation=False,merge_language=False):
        super(ResNet50Conv5ROIFeatureExtractor, self).__init__()
        if cfg.MODEL.ROI_RELATION_HEAD.LM_MULTI_LAYERS and merge_language==True:
            resolution = cfg.MODEL.ROI_RELATION_HEAD.POOLER_RESOLUTION
            scales = cfg.MODEL.ROI_RELATION_HEAD.POOLER_SCALES
            sampling_ratio = cfg.MODEL.ROI_RELATION_HEAD.POOLER_SAMPLING_RATIO
        else:
            resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
            scales = cfg.MODEL.ROI_BOX_HEAD.POOLER_SCALES
            sampling_ratio = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler = Pooler(
            output_size=(resolution, resolution),
            scales=scales,
            sampling_ratio=sampling_ratio,
            in_channels=in_channels,
            cat_all_levels=cat_all_levels
        )

        stage = resnet.StageSpec(index=4, block_count=3, return_features=False)
        head = resnet.ResNetHead(
            block_module=cfg.MODEL.RESNETS.TRANS_FUNC,
            stages=(stage,),
            num_groups=cfg.MODEL.RESNETS.NUM_GROUPS,
            width_per_group=cfg.MODEL.RESNETS.WIDTH_PER_GROUP,
            stride_in_1x1=cfg.MODEL.RESNETS.STRIDE_IN_1X1,
            stride_init=None,
            res2_out_channels=cfg.MODEL.RESNETS.RES2_OUT_CHANNELS,
            dilation=cfg.MODEL.RESNETS.RES5_DILATION
        )

        self.pooler = pooler
        self.head = head
        self.out_channels = head.out_channels

        if cfg.MODEL.RELATION_ON:
            # for the following relation head, the features need to be flattened
            pooling_size = 2
            self.adptive_pool = nn.AdaptiveAvgPool2d((pooling_size, pooling_size))
            input_size = self.out_channels * pooling_size ** 2
            representation_size = cfg.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM
            use_gn = cfg.MODEL.ROI_BOX_HEAD.USE_GN

            if half_out:
                out_dim = int(representation_size / 2)
            else:
                out_dim = representation_size

            self.fc7 = make_fc(input_size, out_dim, use_gn)
            self.resize_channels = input_size
            self.flatten_out_channels = out_dim
            if cfg.MODEL.ROI_RELATION_HEAD.VISUAL_LANGUAGE_MERGER_OBJ and for_relation:
                self.visual_language_merger_obj = make_visual_language_merger_obj(
                    cfg)  # language和visual融合
            self.cfg = cfg
            self.for_relation = for_relation

    def forward(self, x, proposals):
        x = self.pooler(x, proposals)
        x = self.head(x)

        return x#torch.Size([20131, 2048, 4, 4])

    def forward_without_pool(self, x):
        x = self.head(x)
        return self.flatten_roi_features(x)

    def flatten_roi_features(self, x):
        x = self.adptive_pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc7(x))
        return x


@registry.ROI_BOX_FEATURE_EXTRACTORS.register("FPN2MLPFeatureExtractor")
class FPN2MLPFeatureExtractor(nn.Module):
    """
    Heads for FPN for classification
    """

    def __init__(self, cfg, in_channels, half_out=False, cat_all_levels=False, for_relation=False,merge_language=False):
        super(FPN2MLPFeatureExtractor, self).__init__()
        if cfg.MODEL.ROI_RELATION_HEAD.LM_MULTI_LAYERS and merge_language==True:
            resolution = cfg.MODEL.ROI_RELATION_HEAD.POOLER_RESOLUTION
            scales = cfg.MODEL.ROI_RELATION_HEAD.POOLER_SCALES
            sampling_ratio = cfg.MODEL.ROI_RELATION_HEAD.POOLER_SAMPLING_RATIO
        else:
            resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
            scales = cfg.MODEL.ROI_BOX_HEAD.POOLER_SCALES
            sampling_ratio = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler = Pooler(
            output_size=(resolution, resolution),
            scales=scales,
            sampling_ratio=sampling_ratio,
            in_channels=in_channels,
            cat_all_levels=cat_all_levels,
        )
        input_size = in_channels * resolution ** 2
        representation_size = cfg.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM
        use_gn = cfg.MODEL.ROI_BOX_HEAD.USE_GN
        self.pooler = pooler


        if half_out:
            out_dim = int(representation_size / 2)
        else:
            out_dim = representation_size
        if merge_language==False:
            self.fc6 = make_fc(input_size, representation_size, use_gn)
            self.fc7 = make_fc(representation_size, out_dim, use_gn)#512是transformer
            self.resize_channels = input_size
            self.out_channels = out_dim
            self.for_relation = for_relation
            if cfg.MODEL.ROI_RELATION_HEAD.VISUAL_LANGUAGE_MERGER_OBJ and for_relation:
                self.visual_language_merger_obj = make_visual_language_merger_obj(
                cfg)   # language和visual融合
            self.cfg=cfg
    def forward(self, x, proposals):
        x = self.pooler(x, proposals)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc6(x))#4096
        x = F.relu(self.fc7(x))
        if self.cfg.MODEL.ROI_RELATION_HEAD.VISUAL_LANGUAGE_MERGER_OBJ and self.for_relation:
            x=self.visual_language_merger_obj(x, proposals)

        return x

    def forward_without_pool(self, x):#给union feature用的
        b=x.size(0)
        x = x.reshape(b, -1)
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        return x

@registry.ROI_BOX_FEATURE_EXTRACTORS.register("FPNXconv1fcFeatureExtractor")
class FPNXconv1fcFeatureExtractor(nn.Module):
    """
    Heads for FPN for classification
    """

    def __init__(self, cfg, in_channels):
        super(FPNXconv1fcFeatureExtractor, self).__init__()

        resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        scales = cfg.MODEL.ROI_BOX_HEAD.POOLER_SCALES
        sampling_ratio = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler = Pooler(
            output_size=(resolution, resolution),
            scales=scales,
            sampling_ratio=sampling_ratio,
        )
        self.pooler = pooler

        use_gn = cfg.MODEL.ROI_BOX_HEAD.USE_GN
        conv_head_dim = cfg.MODEL.ROI_BOX_HEAD.CONV_HEAD_DIM
        num_stacked_convs = cfg.MODEL.ROI_BOX_HEAD.NUM_STACKED_CONVS
        dilation = cfg.MODEL.ROI_BOX_HEAD.DILATION

        xconvs = []
        for ix in range(num_stacked_convs):
            xconvs.append(
                nn.Conv2d(
                    in_channels,
                    conv_head_dim,
                    kernel_size=3,
                    stride=1,
                    padding=dilation,
                    dilation=dilation,
                    bias=False if use_gn else True
                )
            )
            in_channels = conv_head_dim
            if use_gn:
                xconvs.append(group_norm(in_channels))
            xconvs.append(nn.ReLU(inplace=True))

        self.add_module("xconvs", nn.Sequential(*xconvs))
        for modules in [self.xconvs, ]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    if not use_gn:
                        torch.nn.init.constant_(l.bias, 0)

        input_size = conv_head_dim * resolution ** 2
        representation_size = cfg.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM
        self.fc6 = make_fc(input_size, representation_size, use_gn=False)
        self.out_channels = representation_size

    def forward(self, x, proposals):
        x = self.pooler(x, proposals)
        x = self.xconvs(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc6(x))
        return x


def make_roi_box_feature_extractor(cfg, in_channels, half_out=False, cat_all_levels=False, for_relation=False,merge_language=False):
    func = registry.ROI_BOX_FEATURE_EXTRACTORS[
        cfg.MODEL.ROI_BOX_HEAD.FEATURE_EXTRACTOR
    ]
    return func(cfg, in_channels, half_out, cat_all_levels, for_relation,merge_language)
def make_relation_box_feature_extractor(cfg, in_channels,merge_language=True):
    func = registry.ROI_BOX_FEATURE_EXTRACTORS[
        cfg.MODEL.ROI_RELATION_HEAD.BOX_FEATURE_EXTRACTOR

    ]
    return func(cfg, in_channels,merge_language=merge_language)
class make_visual_language_merger_obj(nn.Module):
    def __init__(self,cfg):
        super(make_visual_language_merger_obj, self).__init__()
        self.cfg=cfg
        self.latest_fusion=False#在决策层相加
        self.early_fusion = True  #
        # statistics = get_dataset_statistics(cfg)
        # obj_classes, rel_classes = statistics['obj_classes'], statistics['rel_classes']
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
        self.use_cls_tocken = self.cfg.MODEL.ROI_RELATION_HEAD.USE_CLS_TOCKEN
        if self.use_cls_tocken:
            self.num_obj_classes=self.num_obj_classes+1
            self.obj_classes.append('[cls]')#拼在最后一位
            self.cls_pos_embed=nn.Embedding(1,128)

        #transformer参数

        self.embed_dim = self.cfg.MODEL.ROI_RELATION_HEAD.EMBED_DIM
        self.hidden_dim = self.cfg.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.nms_thresh = self.cfg.TEST.RELATION.LATER_NMS_PREDICTION_THRES

        self.dropout_rate = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.DROPOUT_RATE
        self.obj_layer = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.OBJ_LAYER
        self.edge_layer = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.REL_LAYER  #todo 沒用上 2
        self.num_head = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.NUM_HEAD
        self.inner_dim = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.INNER_DIM
        self.k_dim = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.KEY_DIM
        self.v_dim = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.VAL_DIM


        self.context_obj = TransformerEncoder(self.obj_layer, self.num_head, self.k_dim,
                                              self.v_dim, self.hidden_dim, self.inner_dim, self.dropout_rate)
        self.bbox_embed = nn.Sequential(*[
            nn.Linear(9, 32), nn.ReLU(inplace=True), nn.Dropout(0.1),
            nn.Linear(32, 128), nn.ReLU(inplace=True), nn.Dropout(0.1),
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
        self.languageembedding = nn.Embedding(self.num_obj_classes,embed_dim)
        # position embedding
        self.pos_embed = nn.Sequential(*[
            nn.Linear(9, 32), nn.BatchNorm1d(32, momentum=0.001),
            nn.Linear(32, 128), nn.ReLU(inplace=True),
        ])
        self.lin_obj = nn.Linear(embed_dim  + 128, self.hidden_dim)
        with torch.no_grad():
            self.languageembedding.weight.copy_(obj_embed_vecs, non_blocking=True)





    def forward(self, visual_feature, proposals):

        if self.early_fusion:
            if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:  # obj_embed_by_pred_dist dim:200。obj_labels是把batch里的拼在一起
                obj_labels = torch.cat([proposal.get_field("labels") for proposal in proposals], dim=0).detach()
                wordembedding_corpus = self.languageembedding(
                    obj_labels.long())  # word embedding层，输入word标签得embedding

            else:
                if self.cfg.MODEL.ROI_RELATION_HEAD.use_possibility_merger:

                    obj_logits = torch.cat([proposal.get_field("predict_logits") for proposal in proposals], dim=0).detach()
                    wordembedding_corpus = obj_logits @ self.languageembedding.weight  # ?
                else:
                    obj_labels = torch.cat([proposal.get_field("pred_labels") for proposal in proposals], dim=0).detach()
                    wordembedding_corpus =self.languageembedding(obj_labels.long())

        pos_embed = self.pos_embed(encode_box_info(proposals))
        num_objs = [len(p) for p in proposals]
        pos_embeds= pos_embed.split(num_objs, dim=0)
        wordembedding_corpus = wordembedding_corpus.split(num_objs, dim=0)
        if self.use_cls_tocken:
            num_objs= list(map(lambda x:x+1,num_objs)) #每个proposal数量加1，即[cls]
            cls_embedding = self.languageembedding(torch.tensor(self.num_obj_classes - 1).cuda()).unsqueeze(0)
        obj_feats=[]
        for wordembedding,pos_embed, proposal in zip(wordembedding_corpus,pos_embeds,proposals):
            # encode objects with transformer
            obj_pre_rep = torch.cat((wordembedding, pos_embed),-1)
            if self.use_cls_tocken:
                obj_pre_rep = torch.cat((torch.cat(
                    (cls_embedding, self.cls_pos_embed(torch.tensor(0).cuda()).unsqueeze(0)), -1), obj_pre_rep),
                                      0)  # [cls]拼接到第一位
            obj_feats.append(obj_pre_rep)
        obj_feats = torch.cat(obj_feats, 0)#transformer中，会重新把feature切分成每个image一个list元素的形式

        obj_feats = self.lin_obj(obj_feats)
        obj_feats=self.context_obj(obj_feats, num_objs)  # TransformerEncoder
        if self.use_cls_tocken:
            cls_feature=[]#存储对应符合visual feature的cls feature
            obj_feats = obj_feats.split(num_objs, dim=0)

            for obj_feat,num_obj in zip(obj_feats,num_objs):
                cls_feature.append(obj_feat[0].unsqueeze(0).repeat((num_obj-1), 1))
            cls_feature=torch.cat(cls_feature,0)
            mixed = torch.cat((visual_feature, cls_feature ), 1)#把[cls]的特征拼在visual的后面
        else:
            mixed=torch.cat((visual_feature,obj_feats),1)
        return  mixed


