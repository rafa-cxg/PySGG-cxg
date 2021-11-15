import array
import os
import zipfile
import itertools
import six
import torch
import numpy as np
from six.moves.urllib.request import urlretrieve
from tqdm import tqdm
import sys
from pysgg.modeling.utils import cat
from  torch.multiprocessing import  Pool
from functools import partial
def normalize_sigmoid_logits(orig_logits):
    orig_logits = torch.sigmoid(orig_logits)
    orig_logits = orig_logits / (orig_logits.sum(1).unsqueeze(-1) + 1e-12)
    return orig_logits

def generate_attributes_target(attributes, device, max_num_attri, num_attri_cat):
        """
        from list of attribute indexs to [1,0,1,0,0,1] form
        """
        assert max_num_attri == attributes.shape[1]
        num_obj = attributes.shape[0]

        with_attri_idx = (attributes.sum(-1) > 0).long()
        attribute_targets = torch.zeros((num_obj, num_attri_cat), device=device).float()

        for idx in torch.nonzero(with_attri_idx).squeeze(1).tolist():
            for k in range(max_num_attri):
                att_id = int(attributes[idx, k])
                if att_id == 0:
                    break
                else:
                    attribute_targets[idx, att_id] = 1
        return attribute_targets, with_attri_idx

def transpose_packed_sequence_inds(lengths):
    """
    Get a TxB indices from sorted lengths. 
    Fetch new_inds, split by new_lens, padding to max(new_lens), and stack.
    Returns:
        new_inds (np.array) [sum(lengths), ]
        new_lens (list(np.array)): number of elements of each time step, descending
    """
    new_inds = []
    new_lens = []
    cum_add = np.cumsum([0] + lengths)
    max_len = lengths[0]
    length_pointer = len(lengths) - 1
    for i in range(max_len):
        while length_pointer > 0 and lengths[length_pointer] <= i:
            length_pointer -= 1
        new_inds.append(cum_add[:(length_pointer+1)].copy())
        cum_add[:(length_pointer+1)] += 1
        new_lens.append(length_pointer+1)
    new_inds = np.concatenate(new_inds, 0)
    return new_inds, new_lens


def sort_by_score(proposals, scores):
    """
    We'll sort everything scorewise from Hi->low, BUT we need to keep images together
    and sort LSTM from l
    :param im_inds: Which im we're on
    :param scores: Goodness ranging between [0, 1]. Higher numbers come FIRST
    :return: Permutation to put everything in the right order for the LSTM
             Inverse permutation
             Lengths for the TxB packed sequence.
    """
    num_rois = [len(b) for b in proposals]
    num_im = len(num_rois)

    scores = scores.split(num_rois, dim=0)
    ordered_scores = []
    for i, (score, num_roi) in enumerate(zip(scores, num_rois)):
        ordered_scores.append( score - 2.0 * float(num_roi * 2 * num_im + i) )
    ordered_scores = cat(ordered_scores, dim=0)
    _, perm = torch.sort(ordered_scores, 0, descending=True)

    num_rois = sorted(num_rois, reverse=True)
    inds, ls_transposed = transpose_packed_sequence_inds(num_rois)  # move it to TxB form
    inds = torch.LongTensor(inds).to(scores[0].device)
    ls_transposed = torch.LongTensor(ls_transposed)
    
    perm = perm[inds] # (batch_num_box, )
    _, inv_perm = torch.sort(perm)

    return perm, inv_perm, ls_transposed


def to_onehot(vec, num_classes, fill=1000):
    """
    Creates a [size, num_classes] torch FloatTensor where
    one_hot[i, vec[i]] = fill
    
    :param vec: 1d torch tensor
    :param num_classes: int
    :param fill: value that we want + and - things to be.
    :return: 
    """
    onehot_result = vec.new(vec.size(0), num_classes).float().fill_(-fill)
    arange_inds = vec.new(vec.size(0)).long()
    torch.arange(0, vec.size(0), out=arange_inds)

    onehot_result.view(-1)[vec.long() + num_classes*arange_inds] = fill
    return onehot_result



def get_dropout_mask(dropout_probability, tensor_shape, device):
    """
    once get, it is fixed all the time
    """
    binary_mask = (torch.rand(tensor_shape) > dropout_probability)
    # Scale mask by 1/keep_prob to preserve output statistics.
    dropout_mask = binary_mask.float().to(device).div(1.0 - dropout_probability)
    return dropout_mask

def center_x(proposals):
    assert proposals[0].mode == 'xyxy'
    boxes = cat([p.bbox for p in proposals], dim=0)
    c_x = 0.5 * (boxes[:, 0] + boxes[:, 2])
    return c_x.view(-1)
    
def encode_box_info(proposals):
    """
    encode proposed box information (x1, y1, x2, y2) to 
    (cx/wid, cy/hei, w/wid, h/hei, x1/wid, y1/hei, x2/wid, y2/hei, wh/wid*hei)
    """
    assert proposals[0].mode == 'xyxy'
    boxes_info = []

    for proposal in proposals:
        boxes = proposal.bbox
        img_size = proposal.size
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
        boxes_info.append(info)
        center=torch.cat([x.int(),y.int()],1)
        proposal.add_field('center',center)

    return torch.cat(boxes_info, dim=0)
def encode_rel_box_info(proposals,rel_pair_idxs):#todo 和上面的函数有很多重叠运算需要优化
    """
    encode proposed box information (x1, y1, x2, y2) to
    (cx/wid, cy/hei, w/wid, h/hei, x1/wid, y1/hei, x2/wid, y2/hei, wh/wid*hei)
    """
    assert proposals[0].mode == 'xyxy'
    boxes_info = []
    for proposal,rel_pair_idx in zip(proposals,rel_pair_idxs):
        sub_idx=rel_pair_idx[:,0]
        obj_idx = rel_pair_idx[:,1]
        boxes = proposal[sub_idx].bbox
        img_size = proposal[sub_idx].size
        wid = img_size[0]
        hei = img_size[1]
        wh = boxes[:, 2:] - boxes[:, :2] + 1.0
        xy = boxes[:, :2] + 0.5 * wh
        sub_w, sub_h = wh.split([1,1], dim=-1)
        sub_x, sub_y = xy.split([1,1], dim=-1)
        sub_x1, sub_y1, sub_x2, sub_y2 = boxes.split([1,1,1,1], dim=-1)
        assert wid * hei != 0
        '''obj_box'''
        boxes = proposal[obj_idx].bbox
        img_size = proposal[obj_idx].size
        wid = img_size[0]
        hei = img_size[1]
        wh = boxes[:, 2:] - boxes[:, :2] + 1.0
        xy = boxes[:, :2] + 0.5 * wh
        obj_w, obj_h = wh.split([1, 1], dim=-1)
        obj_x, obj_y = xy.split([1, 1], dim=-1)
        obj_x1, obj_y1, obj_x2, obj_y2 = boxes.split([1, 1, 1, 1], dim=-1)
        assert wid * hei != 0
        distance=torch.pow(( torch.pow((sub_x-obj_x)/wid,2)+torch.pow((sub_y-obj_y)/hei,2)),0.5)
        iou=(sub_x2-obj_x1)*(sub_y2-obj_y1)/(obj_h*obj_w+sub_h*sub_w-(sub_x2-obj_x1)*(sub_y2-obj_y1))

        info = torch.cat([distance,iou], dim=-1).view(-1, 2)
        boxes_info.append(info)

    return torch.cat(boxes_info, dim=0)

# class PositionalEncoding(nn.Module):
#
#     def __init__(self, d_hid, n_position=200):
#         super(PositionalEncoding, self).__init__()
#
#         # Not a parameter
#         self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))
#
#     def _get_sinusoid_encoding_table(self, n_position, d_hid):
#         ''' Sinusoid position encoding table '''
#         # TODO: make it with torch instead of numpy
#
#         def get_position_angle_vec(position):
#             return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]
#
#         sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
#         sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
#         sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
#
#         return torch.FloatTensor(sinusoid_table).unsqueeze(0)
#
#     def forward(self, x):
#         return x + self.pos_table[:, :x.size(1)].clone().detach()
def env_pos_rel_box_info(proposals,rel_pair_idxs):
    sincos_embed=[]
    BS = torch.nn.BatchNorm1d(4096,momentum=0.001)
    # for proposal,rel_pair_idx in (zip(proposals,rel_pair_idxs)):
    #     # sub_idx=rel_pair_idx[0]
    #     # BS = torch.nn.BatchNorm1d(len(proposal), momentum=0.001)
    #     env_prop_idx=torch.arange(0,len(proposal))
    #     # center_all=torch.cat((proposal.get_field('center')[0],proposal.get_field('center')[1]),dim=1)
    #     center_all=proposal.get_field('center')
    #     grid_x=(center_all[:,0].float().unsqueeze(-1)-center_all[:,0].float().unsqueeze(-1).permute(1,0))*10/proposal.size[0]
    #     grid_y = (center_all[:, 1].float().unsqueeze(-1) - center_all[:, 1].float().unsqueeze(-1).permute(1, 0))*10/proposal.size[1]
    #     # grid=torch.stack((grid_x,grid_y),dim=-1)
    #     del center_all
    #     embed_x=_get_sinusoid_encoding_table(grid_x[rel_pair_idx[:,0]],(4096+128+200))#[1,num_rel,num_prop,200]
    #     del grid_x
    #     embed_y=_get_sinusoid_encoding_table(grid_y[rel_pair_idx[:,0]],(4096+128+200))
    #
    #     del grid_y
    #     embed=torch.cat((embed_x,embed_y),-1)
    #     del embed_x
    #     del embed_y
    #     # bs_embed=BS(embed)
    #     sincos_embed.append(embed)
    #     del embed
    for proposal,rel_pair_idx in (zip(proposals,rel_pair_idxs)):
        center_all = proposal.get_field('center')
        num_obj=torch.arange(center_all.size(0))
        grid_x = (center_all[:, 0].float().unsqueeze(-1) - center_all[:, 0].float().unsqueeze(-1).permute(1, 0)) * 10 /proposal.size[0]
        grid_y = (center_all[:, 1].float().unsqueeze(-1) - center_all[:, 1].float().unsqueeze(-1).permute(1, 0))*10/proposal.size[1]
        embed_x = _get_sinusoid_encoding_table(grid_x[num_obj],(4096 + 128 + 200))  # [1,num_rel,num_prop,200]
        embed_y = _get_sinusoid_encoding_table(grid_y[num_obj], (4096 + 128 + 200))
        obj_mask = rel_pair_idx[:, 1].unsqueeze(-1).repeat(1, (4096 + 200 + 128)).unsqueeze(1)
        single_embed_x=embed_x[rel_pair_idx[:, 0]].gather(1, obj_mask).squeeze(1)
        single_embed_y=embed_y[rel_pair_idx[:, 0]].gather(1, obj_mask).squeeze(1)
        embed = torch.cat((single_embed_x,single_embed_y), -1)#[420,8848]
        sincos_embed.append(embed)
        del embed
        del embed_x,obj_mask
        del embed_y


    return  sincos_embed
def single_process(position,d_hid):#pos:list:[num_sub,5]
    with torch.no_grad():
        # a=torch.zeros([position.size(0),position.size(1),d_hid])#[num_sub,num_prop_200]
        # b = torch.zeros([position.size(0), position.size(1),d_hid]).to('cuda')

        # under=torch.arange(d_hid).to('cuda')
        #
        # for hid_j in range(d_hid):
        #     b[:,:,hid_j]=position[:,:]/ np.power(50, 2 * (hid_j // 2) / d_hid)
        '''torch implecation'''
        hid_j = torch.arange(d_hid, dtype=torch.float16, device='cuda')
        # torch.cuda.memosry_summary(device='cuda', abbreviated=False)
        dom=torch.pow(50, (2 * (hid_j // 2) / d_hid))
        del hid_j
        # b = position[:, :].unsqueeze(-1) /dom
        # del dom
        # assert  a==b
        # for p in position:
        #     a.append(torch.cat([p / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]))

    return position[:, :].unsqueeze(-1) /dom
def get_position_angle_vec(group_input_dim,position,d_hid,indexes):#pos:list:[num_sub,5]
    # a=torch.zeros([position.size(0),position.size(1),d_hid])#[num_sub,num_prop_200]
    b = torch.zeros([position.size(0), position.size(1),group_input_dim[1].size(-1)])
    start=int(indexes[group_input_dim[0]])
    end=int(indexes[group_input_dim[0]]+group_input_dim[1].size(-1))
    for hid_j in range(start,end):
        b[:,:,(hid_j-start)]=position[:,:]/ np.power(50, 2 * (hid_j // 2) / d_hid)
    # assert  a==b
    # for p in position:
    #     a.append(torch.cat([p / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]))
    return b
def _get_sinusoid_encoding_table(position, d_hid):#4424
    ''' Sinusoid position encoding table '''

    # TODO: make it with torch instead of numpy



    sinusoid_table = single_process(position,d_hid)
    # sinusoid_table = multi_process(position,d_hid)
    # sinusoid_table=torch.cat(sinusoid_table,-1)
    del position
    # sinusoid_table = ([get_position_angle_vec(pos_i) for pos_i in position])
    sinusoid_table[:,:, 0::2] = torch.sin(sinusoid_table[:, :,0::2])  # dim 2i
    sinusoid_table[:, :,1::2] = torch.cos(sinusoid_table[:,:, 1::2])  # dim 2i+1
    # torch.cuda.empty_cache()
    return (sinusoid_table)

def multi_process(position,d_hid,num_process=6):

    a = torch.zeros([position.size(0), position.size(1), d_hid],dtype=torch.long)  # [num_sub,num_prop_200]
    group_input_dim=torch.chunk(a,num_process,-1)
    indexes=torch.zeros((num_process,1)).squeeze(-1)
    for i in range(1,num_process):
        indexes[i] = group_input_dim[i - 1].shape[-1] + indexes[i - 1]  # important
    position.share_memory_()
    indexes.share_memory_()
    pool = Pool(processes=num_process)


    pool_result = pool.map(
        partial(get_position_angle_vec,position=position,d_hid=d_hid,indexes=indexes),enumerate(group_input_dim))

    pool.close()
    pool.join()
    return  pool_result

def obj_edge_vectors(names, wv_dir, wv_type='glove.6B', wv_dim=300):
    wv_dict, wv_arr, wv_size = load_word_vectors(wv_dir, wv_type, wv_dim)

    vectors = torch.Tensor(len(names), wv_dim)#len(names):151,wv_dim=200,vectors:[151,200]
    vectors.normal_(0,1)

    for i, token in enumerate(names):
        wv_index = wv_dict.get(token, None)
        if wv_index is not None:
            vectors[i] = wv_arr[wv_index]
        else:
            # Try the longest word
            lw_token = sorted(token.split(' '), key=lambda x: len(x), reverse=True)[0]
            print("{} -> {} ".format(token, lw_token))
            wv_index = wv_dict.get(lw_token, None)
            if wv_index is not None:
                vectors[i] = wv_arr[wv_index]
            else:
                print("fail on {}".format(token))#'__background__' fail是因为下划线，所以要不要去掉下划线？

    return vectors

def load_word_vectors(root, wv_type, dim):
    """Load word vectors from a path, trying .pt, .txt, and .zip extensions."""
    URL = {
        'glove.42B': 'http://nlp.stanford.edu/data/glove.42B.300d.zip',
        'glove.840B': 'http://nlp.stanford.edu/data/glove.840B.300d.zip',
        'glove.twitter.27B': 'http://nlp.stanford.edu/data/glove.twitter.27B.zip',
        'glove.6B': 'http://nlp.stanford.edu/data/glove.6B.zip',
        }
    if isinstance(dim, int):
        dim = str(dim) + 'd'
    fname = os.path.join(root, wv_type + '.' + dim)

    if os.path.isfile(fname + '.pt'):
        fname_pt = fname + '.pt'
        print('loading word vectors from', fname_pt)
        try:
            return torch.load(fname_pt, map_location=torch.device("cpu"))
        except Exception as e:
            print("Error loading the model from {}{}".format(fname_pt, str(e)))
            sys.exit(-1)
    if os.path.isfile(fname + '.txt'):
        fname_txt = fname + '.txt'
        cm = open(fname_txt, 'rb')
        cm = [line for line in cm]
    elif os.path.basename(wv_type) in URL:
        url = URL[wv_type]
        print('downloading word vectors from {}'.format(url))
        filename = os.path.basename(fname)
        if not os.path.exists(root):
            os.makedirs(root)
        with tqdm(unit='B', unit_scale=True, miniters=1, desc=filename) as t:
            fname, _ = urlretrieve(url, fname, reporthook=reporthook(t))
            with zipfile.ZipFile(fname, "r") as zf:
                print('extracting word vectors into {}'.format(root))
                zf.extractall(root)
        if not os.path.isfile(fname + '.txt'):
            raise RuntimeError('no word vectors of requested dimension found')
        return load_word_vectors(root, wv_type, dim)
    else:
        raise RuntimeError('unable to load word vectors')

    wv_tokens, wv_arr, wv_size = [], array.array('d'), None
    if cm is not None:
        for line in tqdm(range(len(cm)), desc="loading word vectors from {}".format(fname_txt)):
            entries = cm[line].strip().split(b' ')
            word, entries = entries[0], entries[1:]
            if wv_size is None:
                wv_size = len(entries)
            try:
                if isinstance(word, six.binary_type):
                    word = word.decode('utf-8')
            except:
                print('non-UTF8 token', repr(word), 'ignored')
                continue
            wv_arr.extend(float(x) for x in entries)
            wv_tokens.append(word)

    wv_dict = {word: i for i, word in enumerate(wv_tokens)}
    wv_arr = torch.Tensor(wv_arr).view(-1, wv_size)
    ret = (wv_dict, wv_arr, wv_size)
    torch.save(ret, fname + '.pt')
    return ret

def reporthook(t):
    """https://github.com/tqdm/tqdm"""
    last_b = [0]
    def inner(b=1, bsize=1, tsize=None):
        if tsize is not None:
            t.total = tsize
        t.update((b - last_b[0]) * bsize)
        last_b[0] = b
    return inner
