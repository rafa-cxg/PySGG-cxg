# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import ipdb
import torch


from pysgg.modeling.roi_heads.relation_head.model_gpsnet import GPSNetContext
from torch import nn
from torch.nn import functional as F

from pysgg.config import cfg
from pysgg.data import get_dataset_statistics
from pysgg.modeling import registry
from pysgg.modeling.make_layers import make_fc
from pysgg.modeling.roi_heads.relation_head.classifier import build_classifier
from pysgg.modeling.roi_heads.relation_head.model_kern import (
    GGNNRelReason,
    InstanceFeaturesAugments,
    to_onehot,
)
from pysgg.modeling.roi_heads.relation_head.model_msdn import MSDNContext
from pysgg.modeling.roi_heads.relation_head.model_naive import (
    PairwiseFeatureExtractor,
)
from pysgg.modeling.utils import cat
from pysgg.structures.boxlist_ops import squeeze_tensor



class Two_stageContext(nn.Module):
    def __init__(
        self,
        cfg,
        in_channels,
        hidden_dim=1024,
        num_iter=2,
        dropout=False,
        gate_width=128,
        use_kernel_function=False,
    ):
        super(Two_stageContext, self).__init__()
        self.cfg = cfg
        self.hidden_dim = hidden_dim
        self.update_step = num_iter

    def forward(
            self,
            inst_features,#instance_num, pooling_dim
            rel_union_features,
            proposals,
            rel_pair_inds,
            rel_gt_binarys=None,
            logger=None,
    ):

        return (
           torch.cat(inst_features,rel_union_features)
        )




@registry.TWO_STAGE_PREDICTOR.register("TwoStagePredictor")
class TwoStagePredictor(nn.Module):
    def __init__(self, config, in_channels):
        super(TwoStagePredictor, self).__init__()
        self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.num_rel_cls = config.MODEL.TWO_STAGE_HEAD.NUM_REL_GROUP+1
        self.use_bias = config.MODEL.ROI_RELATION_HEAD.FREQUENCY_BAIS

        # mode
        if cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
            if cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
                self.mode = "predcls"
            else:
                self.mode = "sgcls"
        else:
            self.mode = "sgdet"

        assert in_channels is not None
        self.pooling_dim = cfg.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM#2048
        self.word_dim=cfg.MODEL.ROI_RELATION_HEAD.EMBED_DIM
        self.geometry_feat_dim = 128
        self.input_dim = (in_channels+self.word_dim+self.geometry_feat_dim)*2#4096
        if cfg.MODEL.TWO_STAGE_HEAD.PURE_SENMENTIC:
            self.input_dim=2048+1024
        self.hidden_dim = config.MODEL.TWO_STAGE_HEAD.HIDDEN_DIM#4096
        #待定self.context_layer=
        # post classification
        self.rel_classifier1 = build_classifier(self.input_dim, self.hidden_dim)
        self.rel_classifier2 = build_classifier(self.hidden_dim, self.num_rel_cls)
        # self.confidence_score = build_classifier(1024, 1)
        self.LN1=torch.nn.LayerNorm([self.hidden_dim])
        self.LN2 = torch.nn.LayerNorm(51)
        self.sigmoid = nn.Sigmoid()
        self.softmax =nn.Softmax(dim=-1)
        self.relu=nn.LeakyReLU(0.2)
        self.init_classifier_weight()

        # for logging things
        self.forward_time = 0



    def init_classifier_weight(self):
        self.rel_classifier1.reset_parameters()
        self.rel_classifier2.reset_parameters()


    def start_preclser_relpn_pretrain(self):
        self.context_layer.set_pretrain_pre_clser_mode()

    def end_preclser_relpn_pretrain(self):
        self.context_layer.set_pretrain_pre_clser_mode(False)

    def forward(
        self,
        inst_proposals,
        rel_pair_idxs,
        rel_labels,
        rel_binarys,
        roi_features,
        union_features,
        obj_rep,
        embed,
        logger=None,

    ):
        """

        :param inst_proposals:
        :param rel_pair_idxs:
        :param rel_labels:
        :param rel_binarys:
            the box pairs with that match the ground truth [num_prp, num_prp]
        :param roi_features:
        :param union_features:
        :param logger:

        Returns:
            obj_dists (list[Tensor]): logits of object label distribution
            rel_dists (list[Tensor])
            rel_pair_idxs (list[Tensor]): (num_rel, 2) index of subject and object
            union_features (Tensor): (batch_num_rel, context_pooling_dim): visual union feature of each pair
        """
        '''obj_feats:[num_prop_all,512], rel_feats:[128,512]'''
        #填写如何编码
        # rel_feats = self.context_layer(
        #     roi_features, union_features, inst_proposals, rel_pair_idxs, rel_binarys, logger
        # )#[num_all_prop,4096],[num_all_pair,4096],list:boxlist,
        if cfg.MODEL.TWO_STAGE_HEAD.transformer_pos:
            rel_feats=[]
            for idx, rel_pair_idx in enumerate(rel_pair_idxs):

                # obj_mask = rel_pair_idx[:, 1].unsqueeze(-1).repeat(1, (4096+200+128)*2).unsqueeze(1)
                # obj_pos_embed = (embed[idx].to('cuda').gather(1, obj_mask)).squeeze(1) + torch.cat((obj_rep[rel_pair_idx[:,0]],obj_rep[rel_pair_idx[:,1]]),-1)
                obj_pos_embed = embed[idx].to('cuda')
                rel_feats.append(obj_pos_embed)

            rel_feats=torch.cat(rel_feats,0)

        if cfg.MODEL.TWO_STAGE_HEAD.UNION_BOX:
            rel_feats = union_features
        if cfg.MODEL.TWO_STAGE_HEAD.PURE_SENMENTIC:
            rel_feats =obj_rep
            if torch.all(torch.isfinite(rel_feats)) != True:
                print('fuck relationloss')
                # num_objs = [len(b) for b in inst_proposals] torch.nonzero(torch.isfinite(torch.max(obj_rep, -1)[0])==False)
                # rel_feats=rel_feats.split(num_objs,0)
                # for rel_feat,inst_proposal in zip(rel_feats,inst_proposals)
                #     inst_proposals[torch.isfinite(torch.max(rel_feat, -1)[0])]
        # confidence_score=self.relu(self.confidence_score(rel_feats[:,2048:]))#只是用位置编码部分算confidence score
        rel_feats = self.rel_classifier1(rel_feats)#[N,4]
        # rel_feats = self.LN1(rel_feats)
        rel_feats = self.relu(rel_feats)
        rel_cls_logits = self.rel_classifier2(rel_feats)  # [N,4]
        # rel_cls_logits = self.LN2(rel_cls_logits)
        rel_cls_logits=(rel_cls_logits)
        # rel_cls_logits=confidence_score*rel_cls_logits
        rel_cls_logits = self.relu(rel_cls_logits)
        num_objs = [len(b) for b in inst_proposals]

        num_rels = [r.shape[0] for r in rel_pair_idxs]
        assert len(num_rels) == len(num_objs)
        rel_cls_logits = rel_cls_logits.split(num_rels, dim=0)#把混在一起的结果按照每张图rel数量划分



        return  rel_cls_logits

@registry.TWO_STAGE_PREDICTOR.register("TwoStageDISTPredictor")
class TwoStageDISTPredictor(nn.Module):
    def __init__(self, config, in_channels):
        super(TwoStageDISTPredictor, self).__init__()
        self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.num_rel_cls = config.MODEL.TWO_STAGE_HEAD.NUM_REL_GROUP+1
        self.use_bias = config.MODEL.ROI_RELATION_HEAD.FREQUENCY_BAIS

        # mode
        if cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
            if cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
                self.mode = "predcls"
            else:
                self.mode = "sgcls"
        else:
            self.mode = "sgdet"

        assert in_channels is not None
        self.pooling_dim = cfg.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM#2048
        self.word_dim=cfg.MODEL.ROI_RELATION_HEAD.EMBED_DIM
        self.geometry_feat_dim = 128
        self.input_dim = (in_channels+self.word_dim+self.geometry_feat_dim)*2#4096
        if cfg.MODEL.TWO_STAGE_HEAD.PURE_SENMENTIC:
            self.input_dim=2048+1024
        self.hidden_dim = config.MODEL.TWO_STAGE_HEAD.HIDDEN_DIM#4096
        #待定self.context_layer=
        # post classification
        self.rel_classifier1 = build_classifier(self.input_dim, self.hidden_dim)
        self.rel_classifier2 = build_classifier(self.hidden_dim, self.num_rel_cls)
        self. BN1=torch.nn.BatchNorm1d(self.hidden_dim)
        self.BN2 = torch.nn.BatchNorm1d(self.num_rel_cls)
        self.sigmoid = nn.Sigmoid()
        self.relu=nn.LeakyReLU(0.2)
        self.init_classifier_weight()

        # for logging things
        self.forward_time = 0



    def init_classifier_weight(self):
        self.rel_classifier1.reset_parameters()
        self.rel_classifier2.reset_parameters()


    def start_preclser_relpn_pretrain(self):
        self.context_layer.set_pretrain_pre_clser_mode()

    def end_preclser_relpn_pretrain(self):
        self.context_layer.set_pretrain_pre_clser_mode(False)

    def forward(
        self,
        inst_proposals,
        rel_pair_idxs,
        rel_labels,
        rel_binarys,
        roi_features,
        union_features,
        obj_rep,
        embed,
        logger=None,

    ):
        """

        :param inst_proposals:
        :param rel_pair_idxs:
        :param rel_labels:
        :param rel_binarys:
            the box pairs with that match the ground truth [num_prp, num_prp]
        :param roi_features:
        :param union_features:
        :param logger:

        Returns:
            obj_dists (list[Tensor]): logits of object label distribution
            rel_dists (list[Tensor])
            rel_pair_idxs (list[Tensor]): (num_rel, 2) index of subject and object
            union_features (Tensor): (batch_num_rel, context_pooling_dim): visual union feature of each pair
        """
        '''obj_feats:[num_prop_all,512], rel_feats:[128,512]'''
        #填写如何编码
        # rel_feats = self.context_layer(
        #     roi_features, union_features, inst_proposals, rel_pair_idxs, rel_binarys, logger
        # )#[num_all_prop,4096],[num_all_pair,4096],list:boxlist,
        if cfg.MODEL.TWO_STAGE_HEAD.transformer_pos:
            rel_feats=[]
            for idx, rel_pair_idx in enumerate(rel_pair_idxs):

                # obj_mask = rel_pair_idx[:, 1].unsqueeze(-1).repeat(1, (4096+200+128)*2).unsqueeze(1)
                # obj_pos_embed = (embed[idx].to('cuda').gather(1, obj_mask)).squeeze(1) + torch.cat((obj_rep[rel_pair_idx[:,0]],obj_rep[rel_pair_idx[:,1]]),-1)
                obj_pos_embed = embed[idx].to('cuda')
                rel_feats.append(obj_pos_embed)
            rel_feats=torch.cat(rel_feats,0)
        if cfg.MODEL.TWO_STAGE_HEAD.UNION_BOX:
            rel_feats = union_features
        if cfg.MODEL.TWO_STAGE_HEAD.PURE_SENMENTIC:
            rel_feats =obj_rep
        rel_feats = self.rel_classifier1(rel_feats)#[N,4]
        rel_feats = self.BN1(rel_feats)
        rel_feats = self.relu(rel_feats)
        rel_cls_logits = self.rel_classifier2(rel_feats)  # [N,4]
        rel_cls_logits = self.BN2(rel_cls_logits)
        rel_cls_logits=F.softmax(rel_cls_logits)
        num_objs = [len(b) for b in inst_proposals]
        num_rels = [r.shape[0] for r in rel_pair_idxs]
        assert len(num_rels) == len(num_objs)
        rel_cls_logits = rel_cls_logits.split(num_rels, dim=0)#把混在一起的结果按照每张图rel数量划分



        return  rel_cls_logits


def make_Two_Stage_predictor(cfg, in_channels):#4096
    func = registry.TWO_STAGE_PREDICTOR[cfg.MODEL.TWO_STAGE_HEAD.PREDICTOR]
    return func(cfg, in_channels)


class DynamicGraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, num_nodes):  # in_features=out_features=1024
        super(DynamicGraphConvolution, self).__init__()

        self.static_adj = nn.Sequential(
            nn.Conv1d(num_nodes, num_nodes, 1, bias=False),#num_nodes:51
            nn.LeakyReLU(0.2))
        self.static_weight = nn.Sequential(
            nn.Conv1d(in_features, out_features, 1),
            nn.LeakyReLU(0.2))

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.conv_global = nn.Conv1d(in_features, in_features, 1)
        self.bn_global = nn.BatchNorm1d(in_features)
        self.relu = nn.LeakyReLU(0.2)

        self.conv_create_co_mat = nn.Conv1d(in_features * 2, num_nodes, 1)
        self.dynamic_weight = nn.Conv1d(in_features, out_features, 1)

    def forward_static_gcn(self, x):  # static gcn，H = LReLU(AsVWs), x=(B,N,C_in)即V
        x = self.static_adj(x.transpose(1, 2))  # (B,N,N')#todo 这里的id卷积起到a矩阵的作用，岂不是可变的吗？ 解答：不可变指的是，所有照片都使用相同a
        x = self.static_weight(x.transpose(1, 2))  # (B,OUT_FEATURE,N)
        return x  # 原文的H向量集合

    def forward_construct_dynamic_graph(self, x):  # 获得ADJ矩阵 x是原文的H向量集合
        ### Model global representations ###
        x_glb = self.gap(x)#avgpool:torch.Size([3, 128, 51])->[3, 128, 1]
        x_glb = self.conv_global(x_glb)
        x_glb = self.bn_global(x_glb)
        x_glb = self.relu(x_glb)
        x_glb = x_glb.expand(x_glb.size(0), x_glb.size(1), x.size(2))  # x_glb是Hg

        ### Construct the dynamic correlation matrix ###
        x = torch.cat((x_glb, x), dim=1)  # 卷积后的rep和输入拼接:[3,128*2,51]
        dynamic_adj = self.conv_create_co_mat(x)  # 计算动态边:conv+sigmoid
        dynamic_adj = torch.sigmoid(dynamic_adj)#torch.Size([3, 51, 51])
        return dynamic_adj

    def forward_dynamic_gcn(self, x, dynamic_adj):  # A*X*W
        x = torch.matmul(x, dynamic_adj)#x still :[3,128,51]
        x = self.relu(x)
        x = self.dynamic_weight(x)
        x = self.relu(x)
        return x

    def forward(self, x):  # 输入的v向量集合
        """ D-GCN module

        Shape:
        - Input: (B, C_in, N) # C_in: 1024, N: num_classes
        - Output: (B, C_out, N) # C_out: 1024, N: num_classes
        """
        out_static = self.forward_static_gcn(x)#out_static:[3,128,51]
        x = x + out_static  # residual,是原图H所在位置,但这里等式右边是v, #todo out_stastic是h，论文里面没有把它加在一起！
        dynamic_adj = self.forward_construct_dynamic_graph(x)#得到attention graph的可变adj matrix[3,51,51]
        x = self.forward_dynamic_gcn(x, dynamic_adj)
        return x#[3,128,51]
@registry.TWO_STAGE_PREDICTOR.register("ADD_GCN")
class ADD_GCN(nn.Module):
    def __init__(self,cfg, in_dim):
        super(ADD_GCN, self).__init__()
        self.num_classes = cfg.MODEL.TWO_STAGE_HEAD.NUM_REL_GROUP+1
        in_dim=256
        self.fc = nn.Conv2d(in_dim, self.num_classes, (1, 1), bias=False)

        self.conv_transform = nn.Conv2d(256, 128, (1, 1))
        self.relu = nn.LeakyReLU(0.2)

        self.gcn = DynamicGraphConvolution(128+8848, 128+8848, self.num_classes)

        self.mask_mat = nn.Parameter(torch.eye(self.num_classes).float())  # n*n的对角阵
        self.last_linear = nn.Conv1d(128+8848, self.num_classes, 1)#如果用的unionfeature则是4424

        # image normalization
        self.image_normalization_mean = [0.485, 0.456, 0.406]
        self.image_normalization_std = [0.229, 0.224, 0.225]


    def forward_classification_sm(self, x):  # 针对生成num_class个的分数，即sm ()
        """ Get another confident scores {s_m}.

        Shape:
        - Input: (B, C_in, H, W) # C_in: 2048
        - Output: (B, C_out) # C_out: num_classes
        """
        x = self.fc(x)  # conv2d: in_channel->num_class
        x = x.view(x.size(0), x.size(1), -1)  # (B, num_class, 卷积后大小h'w')
        x = x.topk(1, dim=-1)[0].mean(dim=-1)  # 每个class特征中最大值做sm的score(B,num_class)
        return x

    def forward_sam(self, x):  # SAM:Semantic Attention Module
        """ SAM module

        Shape:
        - Input: (B, C_in, H, W) # C_in: 2048
        - Output: (B, C_out, N) # C_out: 1024, N: num_classes
        """
        mask = self.fc(x)  # CLASSFIER:CONV2D(B, n_class, H‘, W’)
        mask = mask.view(mask.size(0), mask.size(1), -1)  # (B,N,(H*W)')  这个是activation map
        mask = torch.sigmoid(mask)
        mask = mask.transpose(1, 2)  # (B,(H*W)'，n)

        x = self.conv_transform(x)  # CONV2d:2048->1024    X->X’的conv
        x = x.view(x.size(0), x.size(1), -1)  # (B, C_out, h'w')
        x = torch.matmul(x, mask)  # B,N,H*W  B,H*W,N  X'与感兴趣区域（mask)相乘  问题？两个矩阵h'w'一样吗
        return x

    def forward_dgcn(self, x):
        x = self.gcn(x)
        return x

    def forward(self, x,**kwargs):


        rel_pair_idxs = kwargs['rel_pair_idxs']
        if kwargs.__contains__('union_features'):
            union_features= kwargs['union_features']#【546，4096】
        obj_rep=kwargs['obj_rep']
        embed=kwargs['embed']

        out1 = self.forward_classification_sm(x)  # classfier和其他，输出sm
        if cfg.MODEL.TWO_STAGE_HEAD.INDIVIDUAL_BOX:
            refine_pair_feature = []

            for idx, rel_pair_idx in enumerate(rel_pair_idxs):
                refine_pair_feature.append(torch.cat((obj_rep[rel_pair_idx[:, 0]], obj_rep[rel_pair_idx[:, 1]]), -1).unsqueeze(-1))
                # sub_mask = rel_pair_idx[:, 0].unsqueeze(-1).repeat(1, 4096).unsqueeze(1)
                # obj_mask = rel_pair_idx[:, 1].unsqueeze(-1).repeat(1, 4096).unsqueeze(1)
                # sub_pos_embed = embed[idx].to('cuda').gather(1, sub_mask)+ obj_rep[rel_pair_idx[:, 0], :4096].unsqueeze(1)
                # obj_pos_embed = (embed[idx].to('cuda').gather(1, obj_mask)) + obj_rep[rel_pair_idx[:, 1], :4096].unsqueeze(1)
                # refine_pair_feature.append(torch.cat((sub_pos_embed.squeeze(), obj_rep[rel_pair_idx[:, 0], 4096:], obj_pos_embed.squeeze(),obj_rep[rel_pair_idx[:, 1], 4096:]), dim=1).unsqueeze(-1))  # 8848

        v = self.forward_sam(x)  # (B, C_out:128, N_class), category representations V,
        z=[]
        for idx,i in enumerate(refine_pair_feature):
            i=i.repeat(1,1,4)
            z.append(self.forward_dgcn(torch.cat((i,v[idx].repeat(i.size(0),1,1)),1)))  # final category representation Z (B, C_out:1024, N_class):[3,128,51]
        # z = v + z)
        if cfg.MODEL.TWO_STAGE_HEAD.UNION_BOX:
            '''将z特征和union feature拼再一起'''
            union_features=union_features.unsqueeze(-1).expand(-1,-1,self.num_classes)#[NUM_REL_PAIR,4096,3]
            num_objs = [len(b) for b in rel_pair_idxs]
            union_features = union_features.split(num_objs, dim=0)#[B,N_PRE_IMAGE_REL,4096,3]
            refine_union_features=[]
            for idx,union_feature in enumerate(union_features):
                refine_union_features.append(torch.cat((union_feature,z[idx].expand(num_objs[idx],-1,-1)),1))#相当于有total relation个graph,graph的节点特征和union_feature拼在一起了。（但我是在gcn之后才拼上去的！）Z变换后：[N_PRE_IMAGE_REL,1024,3]
            refine_union_features=torch.cat((refine_union_features))
            out2 = self.last_linear(refine_union_features)  # 一维卷积：(B, N_rel_pairs,N_class） 即Sr
            mask_mat = self.mask_mat.detach()  # [51,51]
            out2 = (out2 * mask_mat).sum(-1)  # final score [3,51] todo:我猜测。sum不影响数值结果(的确)
            out1 = torch.repeat_interleave(out1, torch.tensor(num_objs).to('cuda'), 0)
            final_score = ((out1 + out2) / 2).split(num_objs, dim=0)
            '''end'''
        #我先只加sub obj的相对位置关系
        if cfg.MODEL.TWO_STAGE_HEAD.INDIVIDUAL_BOX:
            final_score = []
            num_objs = [len(b) for b in rel_pair_idxs]
            out1 = torch.repeat_interleave(out1, torch.tensor(num_objs).to('cuda'), 0)
            out1 = out1.split(num_objs, dim=0)
            for idx,z_ in enumerate(z):
                out2 = self.last_linear(z_)
                out2 =F.leaky_relu(out2)
                mask_mat = self.mask_mat.detach()  # [51,51]
                out2 = (out2 * mask_mat).sum(-1)  # final score [3,51] todo:我猜测。sum不影响数值结果(的确)
                shit=1
                final_score.append((out1[idx]+out2)/2)






        return final_score#Sm Sr

    def get_config_optim(self, lr, lrp):
        small_lr_layers = list(map(id, self.features.parameters()))
        large_lr_layers = filter(lambda p: id(p) not in small_lr_layers, self.parameters())
        return [
            {'params': self.features.parameters(), 'lr': lr * lrp},
            {'params': large_lr_layers, 'lr': lr},
        ]
