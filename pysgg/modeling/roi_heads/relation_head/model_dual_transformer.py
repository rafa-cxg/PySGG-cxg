import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from pysgg.data import get_dataset_statistics
from pysgg.modeling import registry
from pysgg.modeling.roi_heads.relation_head.model_cross_transformer import CrossTransformerEncoder
from pysgg.modeling.roi_heads.relation_head.model_transformer import TransformerContext, \
    TransformerEncoder
from pysgg.modeling.utils import cat
from .RTPB.bias_module import build_bias_module
from .utils_motifs import obj_edge_vectors, to_onehot, encode_box_info
from .utils_relation import nms_overlaps
from .utils_relation import layer_init


class GTransformerContext(nn.Module):
    """
        contextual encoding of objects
    """

    def __init__(self, config, obj_classes, rel_classes, in_channels):
        super(GTransformerContext, self).__init__()
        self.cfg = config
        # setting parameters
        if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
            self.mode = 'predcls' if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL else 'sgcls'
        else:
            self.mode = 'sgdet'
        self.obj_classes = obj_classes
        self.rel_classes = rel_classes
        self.num_obj_cls = len(obj_classes)
        self.num_rel_cls = len(rel_classes)
        self.in_channels = in_channels
        self.obj_dim = in_channels
        self.embed_dim = self.cfg.MODEL.ROI_RELATION_HEAD.EMBED_DIM
        self.hidden_dim = self.cfg.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.nms_thresh = self.cfg.TEST.RELATION.LATER_NMS_PREDICTION_THRES

        self.dropout_rate = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.DROPOUT_RATE
        self.obj_layer = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.OBJ_LAYER
        self.num_head = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.NUM_HEAD
        self.inner_dim = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.INNER_DIM
        self.k_dim = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.KEY_DIM
        self.v_dim = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.VAL_DIM

        embed_vecs = obj_edge_vectors(self.obj_classes, wv_dir=self.cfg.GLOVE_DIR, wv_dim=self.embed_dim)
        self.obj_embed1 = nn.Embedding(self.num_obj_cls, self.embed_dim)
        self.obj_embed2 = nn.Embedding(self.num_obj_cls, self.embed_dim)
        with torch.no_grad():
            self.obj_embed1.weight.copy_(embed_vecs, non_blocking=True)
            self.obj_embed2.weight.copy_(embed_vecs, non_blocking=True)

        # position embedding
        self.bbox_embed = nn.Sequential(*[
            nn.Linear(9, 32), nn.ReLU(inplace=True), nn.Dropout(0.1),
            nn.Linear(32, 128), nn.ReLU(inplace=True), nn.Dropout(0.1),
        ])

        embed_dim = self.embed_dim
        # for other embed operation
        if self.cfg.MODEL.ROI_RELATION_HEAD.VISUAL_LANGUAGE_MERGER_OBJ:
            self.language_obj_dim=512
        else: self.language_obj_dim=0
        # ###
        self.lin_obj = nn.Linear(self.in_channels + embed_dim + 128+ self.language_obj_dim, self.hidden_dim)
        layer_init(self.lin_obj, xavier=True)

        self.out_obj = nn.Linear(self.hidden_dim, self.num_obj_cls)
        self.context_obj = TransformerEncoder(self.obj_layer, self.num_head, self.k_dim,
                                              self.v_dim, self.hidden_dim, self.inner_dim, self.dropout_rate)

    def forward(self, roi_features, proposals, rel_pair_idxs=None, logger=None, ctx_average=False):
        # labels will be used in DecoderRNN during training
        use_gt_label = self.training or self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL
        obj_labels = cat([proposal.get_field("labels") for proposal in proposals], dim=0) if use_gt_label else None

        # label/logits embedding will be used as input
        # obj_pred will be use as predicated label
        if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
            obj_embed = self.obj_embed1(obj_labels.long())
            obj_pred = obj_labels
        else:
            obj_logits = cat([proposal.get_field("predict_logits") for proposal in proposals], dim=0).detach()
            obj_embed = F.softmax(obj_logits, dim=1) @ self.obj_embed1.weight
            obj_pred = obj_logits[:, 1:].max(1)[1] + 1

        # bbox embedding will be used as input
        # 'xyxy' --> dim-9 --> fc*2 + ReLU --> dim-128
        assert proposals[0].mode == 'xyxy'
        pos_embed = self.bbox_embed(encode_box_info(proposals))

        # encode objects with transformer
        obj_pre_rep = cat((roi_features, obj_embed, pos_embed), -1)

        num_objs = [len(p) for p in proposals]
        obj_pre_rep = self.lin_obj(obj_pre_rep)


        obj_feats = self.context_obj(obj_pre_rep, num_objs)

        # predict obj_dists and obj_preds
        if self.mode == 'predcls':
            assert obj_labels is not None
            obj_preds = obj_labels
            obj_dists = to_onehot(obj_preds, self.num_obj_cls)
            # edge_pre_rep = cat((roi_features, obj_feats, self.obj_embed2(obj_labels.long())), dim=-1)
        else:
            obj_dists = self.out_obj(obj_feats)
            use_decoder_nms = self.mode == 'sgdet' and not self.training
            if use_decoder_nms:
                boxes_per_cls = [proposal.get_field('boxes_per_cls') for proposal in proposals]
                obj_preds = self.nms_per_cls(obj_dists, boxes_per_cls, num_objs)
            else:
                obj_preds = obj_dists[:, 1:].max(1)[1] + 1
            # edge_pre_rep = cat((roi_features, obj_feats, self.obj_embed2(obj_preds)), dim=-1)

        return obj_dists, obj_preds, obj_feats, None

    def build_sub_graph_mask(self, obj_labels, num_obj):
        batch_size = len(num_obj)
        padding_size = max(num_obj)
        if self.use_weighted_graph_mask:
            res = np.ndarray((batch_size, padding_size, padding_size),
                             dtype=np.float32)  # batch_size * max_obj_cnt * max_obj_cnt
            res[:, :, :] = -1
            start_index = 0
            for img_idx in range(len(num_obj)):
                img_obj_cnt = num_obj[img_idx]
                for i in range(padding_size):
                    res[img_idx, i, i] = 1
                for i in range(start_index, start_index + img_obj_cnt):
                    for j in range(start_index, start_index + img_obj_cnt):
                        if i == j:
                            continue
                        res[img_idx, i - start_index, j - start_index] = self.graph_mask[obj_labels[i]][
                            obj_labels[j]].item()
                start_index += img_obj_cnt
            res = torch.tensor(res, device=obj_labels.device)
            res = F.softmax(res, dim=1)
            return res
        else:
            res = np.ndarray((batch_size, padding_size, padding_size),
                             dtype=np.bool)  # batch_size * max_obj_cnt * max_obj_cnt
            res[:, :, :] = False

            start_index = 0
            for img_idx in range(len(num_obj)):
                img_obj_cnt = num_obj[img_idx]
                for i in range(padding_size):
                    res[img_idx, i, i] = True
                for i in range(start_index, start_index + img_obj_cnt):
                    for j in range(start_index, start_index + img_obj_cnt):
                        if i == j:
                            continue
                        res[img_idx, i - start_index, j - start_index] = self.graph_mask[obj_labels[i]][
                            obj_labels[j]].item()

                start_index += img_obj_cnt
            return torch.tensor(res, device=obj_labels.device)

    def nms_per_cls(self, obj_dists, boxes_per_cls, num_objs):
        obj_dists = obj_dists.split(num_objs, dim=0)
        obj_preds = []
        for i in range(len(num_objs)):
            is_overlap = nms_overlaps(boxes_per_cls[i]).cpu().numpy() >= self.nms_thresh  # (#box, #box, #class)

            out_dists_sampled = F.softmax(obj_dists[i], -1).cpu().numpy()
            out_dists_sampled[:, 0] = -1

            out_label = obj_dists[i].new(num_objs[i]).fill_(0)

            for i in range(num_objs[i]):
                box_ind, cls_ind = np.unravel_index(out_dists_sampled.argmax(), out_dists_sampled.shape)
                out_label[int(box_ind)] = int(cls_ind)
                out_dists_sampled[is_overlap[box_ind, :, cls_ind], cls_ind] = 0.0
                out_dists_sampled[box_ind] = -1.0  # This way we won't re-sample

            obj_preds.append(out_label.long())
        obj_preds = torch.cat(obj_preds, dim=0)
        return obj_preds


class BaseTransformerEncoder(nn.Module):
    def __init__(self, input_dim, out_dim, n_layer, num_head, k_dim, v_dim, dropout_rate=0.1,
                 ):
        super(BaseTransformerEncoder, self).__init__()

        self.dropout_rate = dropout_rate

        self.num_head = num_head
        self.k_dim = k_dim
        self.v_dim = v_dim

        self.graph_encoder = TransformerEncoder(n_layer, self.num_head, self.k_dim,
                                                self.v_dim, input_dim, out_dim, self.dropout_rate)

    def forward(self, features, counts, adj_matrices=None):
        """
        Args:
            features: Feature Tensor to be encoded
            counts: count of item of each sample. [batch-size]
            adj_matrices: None for dense connect.
                List of adjustment matrices with:
                Bool(True for connect) or
                Float(negative for not connected pair)
        Returns:
            Encode result
        """
        if adj_matrices is not None:
            adj_matrices = self.build_padding_adj(adj_matrices, counts)#把adj_matrices每个batch里面的relation数目变成一样多,多出来的部分，保持对角线为1，其余为0
        features = self.graph_encoder(features, counts, adj_matrices)#adj_matrices:[bs,n,n]
        return features

    @staticmethod
    def build_padding_adj(adj_matrices, counts):
        """
        expand the adj matrix to the same size, and stack them into one Tensor
        Args:
            adj_matrices:
            counts:
        Returns:
        """
        padding_size = max(counts)
        index = torch.arange(padding_size).long()

        res = []
        for adj in adj_matrices:
            expand_mat = torch.zeros(size=(padding_size, padding_size)) - 1
            expand_mat[index, index] = 1
            expand_mat = expand_mat.to(adj)
            adj_count = adj.size(0)
            expand_mat[:adj_count, :adj_count] = adj
            res.append(expand_mat)

        return torch.stack(res)


