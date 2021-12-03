# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import bisect
import copy
import json
import logging
import os
import pickle
from collections import Counter
from functools import partial

import numpy as np
import torch
import torch.utils.data
from matplotlib import pyplot as plt
from tqdm import tqdm
import torch.distributed as dist

from pysgg.config import cfg
from pysgg.utils.comm import get_world_size, is_main_process, synchronize
from pysgg.utils.imports import import_file
from pysgg.utils.miscellaneous import save_labels

from . import datasets as D
from . import samplers
from .collate_batch import BatchCollator, BBoxAugCollator
from .transforms import build_transforms

import time
from pysgg.utils.util_from_deepcluster import AverageMeter

from  multiprocessing import  Pool
from scipy.stats import wasserstein_distance
from sklearn_extra.cluster import KMedoids
#by cxg
def compute_Wasserstein_distance(attr_prob,outfile=None):
    # MAX_PROCESS=multiprocessing.cpu_count()
    MAX_PROCESS=40
    cls_num = attr_prob.shape[0]
    similarity = np.zeros((cls_num, cls_num))
    group_attr_prob=np.array_split(attr_prob, MAX_PROCESS)
    indexes=np.zeros((MAX_PROCESS,1))
    for i in range(1,MAX_PROCESS):
        indexes[i]=group_attr_prob[i-1].shape[0]+indexes[i-1]#important
    # multiprocessing.set_start_method('spawn')
    pool = Pool(processes=MAX_PROCESS)

    # arg=[(group_attr_prob,attr_prob,cls_num,similarity)]
    # for line,result in enumerate(tqdm.tqdm(pool.map(partial(multiprocess, original_array=attr_prob,col_number=cls_num,similarity=similarity),group_attr_prob), total=len(group_attr_prob))):
    #     similarity[line, :] = result
    pool_result =pool.map(partial(multiprocess_Wasserstein, original_array=attr_prob,col_number=cls_num,similarity=similarity,indexes=indexes),enumerate(group_attr_prob))
    pool.close()
    pool.join()
    start_line=0
    # fuck=np.zeros((10,10,1234))
    for line, result in enumerate(pool_result):
        similarity[int(indexes[line]):int(indexes[line]+result.shape[0]), :] = result
        start_line=start_line + group_attr_prob[line].shape[0]
    # for line, result in enumerate(fuck):
    #     similarity[line:line+fuck.shape[0], :] = result

    # np.save(outfile, similarity)
    return  similarity
def multiprocess_Wasserstein(group_array_,original_array,col_number,similarity,indexes):
    # for attr in group_array:
    index,group_array=group_array_
    indi_cls_num = group_array.shape[0]
    for i in range(int(indexes[index]), int(indexes[index]+(indi_cls_num))):
        # if i % 50 == 0:
        #     print('had proccessed {} cls...\n'.format(i))
        for j in range(0, int(col_number)):
            if i == j:
                similarity[i, j] = 0
            else:
                 # similarity[i, j] = 0.5 * (cp_kl(original_array[i, :], original_array[j, :])
                 #                          + cp_kl(original_array[j, :], original_array[i, :]))
                 similarity[i, j]=wasserstein_distance(original_array[i,:],original_array[j,:])
    return similarity[int(indexes[index]): int(indexes[index]+(indi_cls_num)),:]
def norm_distribution(prob, num=150,dim=1):
    num=prob.shape[1]
    prob=prob
    prob_weight = prob[:, :num].numpy()
    sum_value = np.sum(prob_weight, keepdims=True, axis=dim)

    prob_weight = prob_weight / np.repeat(sum_value, prob_weight.shape[dim], axis=dim)
    return prob_weight
def plot_distribution_bar(rel_array,data,object=False): #input should be an np array
    stastic_photo_dir='clustering/stastic_photo_dir/'
    CHECK_FOLDER = os.path.isdir(stastic_photo_dir)
    if not CHECK_FOLDER:
        os.makedirs(stastic_photo_dir)
        print("created folder : ", stastic_photo_dir)

    else:
        print(stastic_photo_dir, "folder already exists.")

    for predicates in range(rel_array.shape[0]) :

        objects = (rel_array[predicates,:])#(151)
        xs = np.arange(len(objects))
        fig = plt.figure(figsize=(10, 5))
        plt.bar(xs, objects, color='maroon',width=0.4)
        # plt.xticks(xs, labels)  # Replace default x-ticks with xs, then replace xs with labels
        # plt.yticks(ys)
        if object==False:#代表统计的是predicate
            label = data.ind_to_predicates[predicates]
        else:
            label = data.ind_to_classes[predicates]

        plt.xlabel(label)
        plt.show()
        plt.savefig(stastic_photo_dir+str(label)+'.png')


def map_predicate2cluster(data,cluster_array):
    data_dir='clustering/'
    predicate2cluster = {}
    predicatename2cluster = {}
    for _ in range(0, 3):
        predicate2cluster[str(_)] = [str(j+1) for j in np.where(cluster_array==_)[0]]
        predicatename2cluster[str(_)] = [data.ind_to_predicates[int(i+1)] for i in (np.where(cluster_array==_))[0]]
    with open(os.path.join(data_dir, 'predicatename2cluster.json'), 'w') as ft:
        json.dump(predicatename2cluster, ft)
    with open(os.path.join(data_dir, 'predicate2cluster.json'), 'w') as ft:
        json.dump(predicate2cluster, ft)
    return predicate2cluster, predicatename2cluster


# by Jiaxin
def get_dataset_statistics(cfg):
    """
    get dataset statistics (e.g., frequency bias) from training data
    will be called to help construct FrequencyBias module
    """
    logger = logging.getLogger(__name__)
    logger.info('-' * 100)
    logger.info('get dataset statistics...')
    paths_catalog = import_file(
        "pysgg.config.paths_catalog", cfg.PATHS_CATALOG, True
    )
    DatasetCatalog = paths_catalog.DatasetCatalog
    dataset_names = cfg.DATASETS.TRAIN

    data_statistics_name = ''.join(dataset_names) + '_statistics'
    save_file = os.path.join(cfg.OUTPUT_DIR, "{}.cache".format(data_statistics_name))

    if os.path.exists(save_file):
        logger.info('Loading data statistics from: ' + str(save_file))
        logger.info('-' * 100)
        return torch.load(save_file, map_location=torch.device("cpu"))

    statistics = []
    for dataset_name in dataset_names:
        data = DatasetCatalog.get(dataset_name, cfg)
        factory = getattr(D, data["factory"])
        args = data["args"]
        dataset = factory(**args)
        if "VG_stanford" in dataset_name:
            get_dataset_distribution(dataset, dataset_name)
        statistics.append(dataset.get_statistics())
    logger.info('finish')

    assert len(statistics) == 1
    result = {
        'fg_matrix': statistics[0]['fg_matrix'],
        'pred_dist': statistics[0]['pred_dist'],
        'obj_classes': statistics[0]['obj_classes'],  # must be exactly same for multiple datasets
        'rel_classes': statistics[0]['rel_classes'],
        'att_classes': statistics[0]['att_classes'],
    }
    logger.info('Save data statistics to: ' + str(save_file))
    logger.info('-' * 100)
    torch.save(result, save_file)
    return result

def multi_process_stastics(train_idx,train_data,rel_obj_distribution,sub_rel_distribution,obj_rel_distribution,pred_counter,num_process=10):


    for i in tqdm(train_idx):

        tgt_rel_matrix = train_data.get_groundtruth(i, inner_idx=False).get_field("relation")  # [n*n]
        tgt_pair_idxs = torch.nonzero(tgt_rel_matrix > 0)
        tgt_head_idxs = tgt_pair_idxs[:, 0].contiguous().view(-1)
        tgt_tail_idxs = tgt_pair_idxs[:, 1].contiguous().view(-1)
        tgt_rel_labs = tgt_rel_matrix[tgt_head_idxs, tgt_tail_idxs].contiguous().view(-1)
        sub_class = train_data.get_groundtruth(i, inner_idx=False).get_field("labels")[tgt_head_idxs]
        obj_class = train_data.get_groundtruth(i, inner_idx=False).get_field("labels")[tgt_tail_idxs]
        relation_tuple = torch.stack((sub_class, obj_class, tgt_rel_labs), dim=1)
        for idx, each in enumerate(tgt_rel_labs):
            pred_counter[int(each)] += 1  # 计算的是每个rel出现累计次数
            rel_obj_distribution[int(each)][relation_tuple[idx][0]] += 1
            rel_obj_distribution[int(each)][relation_tuple[idx][1]] += 1
        '''统计图片中出现的所有object'''
        for r in relation_tuple:
            sub_rel_distribution[int(r[0])][int(r[2])] += 1
            obj_rel_distribution[int(r[1])][int(r[2])] += 1
    return  rel_obj_distribution,sub_rel_distribution,obj_rel_distribution,pred_counter

#by cxg
def get_dataset_distribution(train_data, dataset_name,record_rel_distribution=False,clustering=False):
    """save relation frequency distribution after the sampling etc processing
    the data distribution that model will be trained on it

    Args:
        train_data ([type]): [description]
        dataset_name ([type]): [description]
    """
    # 
    if is_main_process():
        print("Get relation class frequency distribution on dataset.")
        pred_counter = Counter()
        rel_obj_distribution = torch.zeros((51,151),dtype=torch.int)
        sub_rel_distribution = torch.zeros((151,51),dtype=torch.int)
        obj_rel_distribution = torch.zeros((151, 51), dtype=torch.int)
        num_process=20
        pool = Pool(processes=num_process)
        split_data = torch.chunk(torch.arange(0,len(train_data),dtype=torch.int), num_process, -1)
        f1 = pool.map(
            partial(multi_process_stastics, train_data=train_data,rel_obj_distribution=rel_obj_distribution, sub_rel_distribution=sub_rel_distribution,obj_rel_distribution=obj_rel_distribution, pred_counter=pred_counter),
            split_data)

        pool.close()
        pool.join()
        '''把多进程内容融合'''
        rel_obj_distribution = torch.zeros((51, 151), dtype=torch.int)
        sub_rel_distribution = torch.zeros((151, 51), dtype=torch.int)
        obj_rel_distribution = torch.zeros((151, 51), dtype=torch.int)
        for part in f1:
            rel_obj_distribution+=part[0]
            sub_rel_distribution+=part[1]
            obj_rel_distribution += part[2]
            pred_counter+=part[3]
        #归一化
        sub_rel_distribution=np.concatenate((norm_distribution(sub_rel_distribution[0:,:-1]),sub_rel_distribution[:,-1]),1)
        obj_rel_distribution =np.concatenate((norm_distribution(obj_rel_distribution[0:,:-1]),obj_rel_distribution[:,-1]),1)
        rel_obj_distribution = norm_distribution(rel_obj_distribution[0:,0:])
        with open(os.path.join(cfg.OUTPUT_DIR, "pred_counter.pkl"), 'wb') as f:
            pickle.dump(pred_counter, f)
        if record_rel_distribution:
            with open(os.path.join(cfg.OUTPUT_DIR, "record_rel_distribution.pkl"), 'wb') as f:
                pickle.dump(rel_obj_distribution, f)
            '''转换成np array'''
            rel_obj_distribution = rel_obj_distribution.numpy()
            # rel_obj_distribution = np.array(rel_obj_distribution)  # (51,151)
            plot_distribution_bar(rel_obj_distribution,train_data)
        '''sub'''
        with open(os.path.join(cfg.OUTPUT_DIR, "record_sub_distribution.pkl"), 'wb') as f:
            pickle.dump(sub_rel_distribution, f)
        plot_distribution_bar(sub_rel_distribution, train_data,object=True)
        '''obj'''
        with open(os.path.join(cfg.OUTPUT_DIR, "record_obj_distribution.pkl"), 'wb') as f:
            pickle.dump(obj_rel_distribution, f)
        plot_distribution_bar(obj_rel_distribution, train_data, object=True)

        if clustering:
            # rel_obj_distribution_norm=norm_distribution(rel_obj_distribution[1:,1:])
            Wasserstein_distance_mat=compute_Wasserstein_distance(rel_obj_distribution[1:,1:])
            kmedoids = KMedoids(n_clusters=3, metric='precomputed', method='pam', init='random', max_iter=100000).fit(
                Wasserstein_distance_mat)
            predicate2cluster,predicatename2cluster = map_predicate2cluster(train_data, kmedoids.labels_)
            a=1







        from pysgg.data.datasets.visual_genome import HEAD, TAIL, BODY
        
        head = HEAD
        body = BODY
        tail = TAIL

        count_sorted = []
        counter_name = []
        cate_set = []
        cls_dict = train_data.ind_to_predicates

        for idx, name_set in enumerate([head, body, tail]):
            # sort the cate names accoding to the frequency
            part_counter = []
            for name in name_set:
                part_counter.append(pred_counter[name])
            part_counter = np.array(part_counter)
            sorted_idx = np.flip(np.argsort(part_counter))

            # reaccumulate the frequency in sorted index
            for j in sorted_idx:
                name = name_set[j]
                cate_set.append(idx)
                counter_name.append(cls_dict[name])
                count_sorted.append(pred_counter[name])

        count_sorted = np.array(count_sorted)

        fig, axs_c = plt.subplots(1, 1, figsize=(16, 5), tight_layout=True)
        palate = ['r', 'g', 'b']
        color = [palate[idx] for idx in cate_set]
        axs_c.bar(counter_name, count_sorted, color=color)
        axs_c.grid()
        plt.xticks(rotation=-60)
        axs_c.set_ylim(0, 50000)
        fig.set_facecolor((1, 1, 1))

        save_file = os.path.join(cfg.OUTPUT_DIR, "rel_freq_dist.png")
        fig.savefig(save_file, dpi=300)
    synchronize()

def compute_features(dataloader, N):
    device = torch.device(cfg.MODEL.DEVICE)
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    with open(os.path.join(cfg.OUTPUT_DIR, "record_sub_distribution.pkl"), 'rb') as f1:
        sub_distribution = torch.tensor(pickle.load(f1)).clone().detach().to('cuda')
    with open(os.path.join(cfg.OUTPUT_DIR, "record_obj_distribution.pkl"), 'rb') as f2:
        obj_distribution = torch.tensor(pickle.load(f2)).clone().detach().to('cuda')
    if dist.get_rank() == 0:
        print('Compute features')
    batch_time = AverageMeter()
    end = time.time()
    features = []
    # discard the label information in the dataloader
    for i, (images, targets, _) in tqdm(enumerate(dataloader),total=int(len(dataloader.dataset)/cfg.SOLVER.IMS_PER_BATCH/num_gpus)):
        # images = images.to(device)
        labels =[target.get_field('labels') for target in targets]
        relation_tuples = [target.get_field('relation_tuple') for target in targets]
        labels=torch.cat(labels)
        relation_tuples=torch.cat(relation_tuples)
        # for label,relation_tuple in zip(labels,relation_tuples):
        sub=sub_distribution[labels[relation_tuples[:, 0]]]
        obj = obj_distribution[labels[relation_tuples[:, 1]]]
        rel = relation_tuples[:, 2].to('cuda').type(dtype=torch.double).unsqueeze(-1)
        aux=torch.cat((sub,obj,rel),-1)



        # aux = aux.astype('float32')
        if i < len(dataloader):
            features.append(aux) #TODO BATCH是整个还是单个gpu的，值得商榷


        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if   (i % 200) == 0:
            print('{0} / {1}\t'
                  'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})'
                  .format(i, len(dataloader), batch_time=batch_time))
    return torch.cat(features,0)
def build_dataset(cfg, dataset_list, transforms, dataset_catalog, is_train=True):
    """
    Arguments:
        dataset_list (list[str]): Contains the names of the datasets, i.e.,
            coco_2014_trian, coco_2014_val, etc
        transforms (callable): transforms to apply to each (image, target) sample
        dataset_catalog (DatasetCatalog): contains the information on how to
            construct a dataset.
        is_train (bool): whether to setup the dataset for training or testing
    """
    if not isinstance(dataset_list, (list, tuple)):
        raise RuntimeError(
            "dataset_list should be a list of strings, got {}".format(dataset_list)
        )
    datasets = []
    for dataset_name in dataset_list:
        data = dataset_catalog.get(dataset_name, cfg)#用name和cfg来确定是否使用filter过滤没有rel的图片
        factory = getattr(D, data["factory"]) #'data["factory"]：VGDataset' 位于visual_genome.py
        args = data["args"]#dictory,有img_dir..anno文件路径
        # for COCODataset, we want to remove images without annotations
        # during training
        if data["factory"] == "COCODataset":
            args["remove_images_without_annotations"] = is_train
        if data["factory"] == "PascalVOCDataset":
            args["use_difficult"] = not is_train
        args["transforms"] = transforms# 包括水平翻转等object
        # make dataset from factory
        # print("build dataset with args:")
        # pprint(args)
        dataset = factory(**args)#通过arg提供vgddataset参数
        datasets.append(dataset)

    # for testing, return a list of datasets
    if not is_train:
        return datasets

    # for training, concatenate all datasets into a single one
    dataset = datasets[0]
    if len(datasets) > 1:
        dataset = D.ConcatDataset(datasets) #dataset子类

    return [dataset]


def make_data_sampler(dataset, shuffle, distributed):
    if distributed:
        return samplers.DistributedSampler(dataset, shuffle=shuffle)
    if shuffle:
        sampler = torch.utils.data.sampler.RandomSampler(dataset)
    else:
        sampler = torch.utils.data.sampler.SequentialSampler(dataset)
    return sampler


def _quantize(x, bins):
    bins = copy.copy(bins)
    bins = sorted(bins)
    quantized = list(map(lambda y: bisect.bisect_right(bins, y), x))#从小到大排序，值一样的排在最右边
    return quantized


def _compute_aspect_ratios(dataset):
    aspect_ratios = []
    for i in range(len(dataset)):
        if hasattr(dataset, "idx_list"):
            i = dataset.idx_list[i]
        img_info = dataset.img_info[i]
        aspect_ratio = float(img_info["height"]) / float(img_info["width"])
        aspect_ratios.append(aspect_ratio)
    return aspect_ratios


def make_batch_data_sampler(
        dataset, sampler, aspect_grouping, images_per_batch, num_iters=None, start_iter=0
):
    if aspect_grouping:
        if not isinstance(aspect_grouping, (list, tuple)):#aspect_grouping是否是list或者tuple。
            aspect_grouping = [aspect_grouping]
        aspect_ratios = _compute_aspect_ratios(dataset)#hight/weight
        group_ids = _quantize(aspect_ratios, aspect_grouping)
        batch_sampler = samplers.GroupedBatchSampler(#todo GroupedBatchSampler和IterationBasedBatchSampler区别
            sampler, group_ids, images_per_batch, drop_uneven=False
        )
    else:
        batch_sampler = torch.utils.data.sampler.BatchSampler(
            sampler, images_per_batch, drop_last=False
        )
    if num_iters is not None:
        batch_sampler = samplers.IterationBasedBatchSampler(#建立在BatchSampler上的
            batch_sampler, num_iters, start_iter
        )
    return batch_sampler


def make_data_loader(cfg, mode='train', is_distributed=False, start_iter=0,for_cluster=False):#for_cluster代表是否是为cluster做的dataloader
    assert mode in {'train', 'val', 'test'}
    num_gpus = get_world_size()
    is_train = mode == 'train'
    if is_train:
        images_per_batch = cfg.SOLVER.IMS_PER_BATCH


        assert (
                images_per_batch % num_gpus == 0
        ), "SOLVER.IMS_PER_BATCH ({}) must be divisible by the number of GPUs ({}) used.".format(
            images_per_batch, num_gpus)
        images_per_gpu = images_per_batch // num_gpus
        shuffle = True
        num_iters = cfg.SOLVER.MAX_ITER
        if for_cluster==True:
            num_iters=None
    else:
        images_per_batch = cfg.TEST.IMS_PER_BATCH
        '''cxg'''
        # if is_distributed:
        #     images_per_batch=cfg.SOLVER.IMS_PER_BATCH*num_gpus
        '''cxg'''
        assert (
                images_per_batch % num_gpus == 0
        ), "TEST.IMS_PER_BATCH ({}) must be divisible by the number of GPUs ({}) used.".format(
            images_per_batch, num_gpus)
        images_per_gpu = images_per_batch // num_gpus
        shuffle = False if not is_distributed else True
        num_iters = None
        start_iter = 0

    # if images_per_gpu > 1:
    #     logger = logging.getLogger(__name__)
    #     logger.warning(
    #         "When using more than one image per GPU you may encounter "
    #         "an out-of-memory (OOM) error if your GPU does not have "
    #         "sufficient memory. If this happens, you can reduce "
    #         "SOLVER.IMS_PER_BATCH (for training) or "
    #         "TEST.IMS_PER_BATCH (for inference). For training, you must "
    #         "also adjust the learning rate and schedule length according "
    #         "to the linear scaling rule. See for example: "
    #         "https://github.com/facebookresearch/Detectron/blob/master/configs/getting_started/tutorial_1gpu_e2e_faster_rcnn_R-50-FPN.yaml#L14"
    #     )

    # group images which have similar aspect ratio. In this case, we only
    # group in two cases: those with width / height > 1, and the other way around,
    # but the code supports more general grouping strategy
    aspect_grouping = [1] if cfg.DATALOADER.ASPECT_RATIO_GROUPING else []

    paths_catalog = import_file(
        "pysgg.config.paths_catalog", cfg.PATHS_CATALOG, True
    )
    DatasetCatalog = paths_catalog.DatasetCatalog
    if mode == 'train':
        dataset_list = cfg.DATASETS.TRAIN
    elif mode == 'val':
        dataset_list = cfg.DATASETS.VAL
    else:
        dataset_list = cfg.DATASETS.TEST

    # If bbox aug is enabled in testing, simply set transforms to None and we will apply transforms later
    transforms = None if not is_train and cfg.TEST.BBOX_AUG.ENABLED else build_transforms(cfg, is_train)
    datasets = build_dataset(cfg, dataset_list, transforms, DatasetCatalog, is_train)

    if is_train:
        # save category_id to label name mapping
        save_labels(datasets, cfg.OUTPUT_DIR)

    data_loaders = []
    for dataset in datasets: #VGDataset
        # print('============')
        # print(len(dataset))
        # print(images_per_gpu)
        # print('============')
        sampler = make_data_sampler(dataset, shuffle, is_distributed)#torch自带sampler
        batch_sampler = make_batch_data_sampler(
            dataset, sampler, aspect_grouping, images_per_gpu, num_iters, start_iter
        )
        collator = BBoxAugCollator() if not is_train and cfg.TEST.BBOX_AUG.ENABLED else \
            BatchCollator(cfg.DATALOADER.SIZE_DIVISIBILITY)
        num_workers = cfg.DATALOADER.NUM_WORKERS
        data_loader = torch.utils.data.DataLoader(
            dataset,
            num_workers=num_workers,
            batch_sampler=batch_sampler,
            collate_fn=collator,
            pin_memory=True
        )
        data_loaders.append(data_loader)
    if is_train:
        # during training, a single (possibly concatenated) data_loader is returned
        assert len(data_loaders) == 1
        return data_loaders[0]
    return data_loaders




