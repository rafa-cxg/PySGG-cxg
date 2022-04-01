# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Basic training script for PyTorch
"""

# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)

import argparse
import datetime
import os
import random
import time
import threading
import json

import gpustat
import numpy as np
import torch
from tqdm import tqdm
import pickle
import torch.multiprocessing as mp

from pysgg.config.defaults import  _C as cfg
from pysgg.data import make_data_loader
from pysgg.engine.inference import inference
from pysgg.engine.trainer import reduce_loss_dict
from pysgg.modeling.detector import build_detection_model
from pysgg.solver import make_lr_scheduler
from pysgg.solver import make_optimizer
from pysgg.utils.checkpoint import DetectronCheckpointer
from pysgg.utils.checkpoint import clip_grad_norm
from pysgg.utils import visualize_graph as vis_graph
from pysgg.utils.collect_env import collect_env_info
from pysgg.utils.comm import synchronize, get_rank, all_gather,get_world_size
from pysgg.utils.logger import setup_logger, debug_print, TFBoardHandler_LEVEL
from pysgg.utils.metric_logger import MetricLogger
from pysgg.utils.miscellaneous import mkdir, save_config
from pysgg.utils.global_buffer import save_buffer
from pysgg.data.build import compute_features
import torch.distributed as dist
from pysgg.utils.comm import all_gather

# from clustering import  clustering
from pysgg.utils.comm import is_main_process
# See if we can use apex.DistributedDataParallel instead of the torch default,
# and enable mixed-precision via apex.amp
try:
    from apex import amp
except ImportError:
    raise ImportError("Use APEX for multi-precision via apex.amp")

SEED = 555#666

torch.cuda.manual_seed(SEED)  # 为当前GPU设置随机种子
torch.cuda.manual_seed_all(SEED)  # 为所有GPU设置随机种子
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

torch.backends.cudnn.enabled = True  # 默认值
torch.backends.cudnn.benchmark = True  # 默认为False
torch.backends.cudnn.deterministic = True  # 默认为False;benchmark为True时,y要排除随机性必须为True

# torch.backends.cudnn.enabled=False  #when train motif
torch.autograd.set_detect_anomaly(True)

SHOW_COMP_GRAPH = False


def show_params_status(model):
    """
    Prints parameters of a model
    """
    st = {}
    strings = []
    total_params = 0
    trainable_params = 0
    for p_name, p in model.named_parameters():

        if not ("bias" in p_name.split(".")[-1] or "bn" in p_name.split(".")[-1]):
            st[p_name] = ([str(x) for x in p.size()], np.prod(p.size()), p.requires_grad)
        total_params += np.prod(p.size())
        if p.requires_grad:
            trainable_params += np.prod(p.size())
    for p_name, (size, prod, p_req_grad) in st.items():
        strings.append(
            "{:<80s}: {:<16s}({:8d}) ({})".format(
                p_name, "[{}]".format(",".join(size)), prod, "grad" if p_req_grad else "    "
            )
        )
    strings = "\n".join(strings)
    return (
        f"\n{strings}\n ----- \n \n"
        f"      trainable parameters:  {trainable_params/ 1e6:.3f}/{total_params / 1e6:.3f} M \n "
    )


def train(
    cfg,
    local_rank,
    distributed,
    logger,
):
    global SHOW_COMP_GRAPH
    output_dir = cfg.OUTPUT_DIR
    debug_print(logger, "prepare training")
    model = build_detection_model(cfg)
    model.train()
    debug_print(logger, "end model construction")
    logger.info(str(model))
    # modules that should be always set in eval mode
    # their eval() method should be called after model.train() is called
    eval_modules = (# tuple
        model.rpn,
        model.backbone,
        model.roi_heads.box,
    )
    train_modules = ()
    rel_pn_module_ref = []
    if cfg.MODEL.ROI_RELATION_HEAD.FIX_FEATURE:#false
        if cfg.MODEL.ROI_RELATION_HEAD.PREDICTOR == "CausalAnalysisPredictor":
            eval_modules = [
                model.rpn,
                model.backbone,
                model.roi_heads.box,
                model.roi_heads.relation.box_feature_extractor,
                model.roi_heads.relation.union_feature_extractor,
                model.roi_heads.relation.predictor.context_layer,
                model.roi_heads.relation.predictor.spt_emb,
            ]
        if cfg.MODEL.ROI_RELATION_HEAD.PREDICTOR == "NaivePredictor":
            eval_modules = [
                model.rpn,
                model.backbone,
                model.roi_heads.box,
                model.roi_heads.relation.box_feature_extractor,
                model.roi_heads.relation.union_feature_extractor,
                model.roi_heads.relation.predictor.obj_pair_feature_extractor,
                model.roi_heads.relation.predictor.pairwise_obj_feat_updim_fc,
                model.roi_heads.relation.predictor.output_fc,
            ]
            if cfg.MODEL.ROI_RELATION_HEAD.CAUSAL.SPATIAL_FOR_VISION:
                eval_modules.append(model.roi_heads.relation.predictor.spt_emb)

        if cfg.MODEL.ROI_RELATION_HEAD.PREDICTOR == "MotifPredictor":
            eval_modules = [
                model.rpn,
                model.backbone,
                model.roi_heads.box,
                model.roi_heads.relation.box_feature_extractor,
                model.roi_heads.relation.union_feature_extractor,
                model.roi_heads.relation.predictor.context_layer,
            ]
        if cfg.MODEL.ROI_RELATION_HEAD.PREDICTOR == "KERNPredictor":
            eval_modules = [
                model.rpn,
                model.backbone,
                model.roi_heads.box,
                model.roi_heads.relation.box_feature_extractor,
                model.roi_heads.relation.union_feature_extractor,
                model.roi_heads.relation.predictor.output_fc,
                model.roi_heads.relation.predictor.obj_pair_feature_extractor,
                model.roi_heads.relation.predictor.pairwise_obj_feat_updim_fc,
                model.roi_heads.relation.predictor.spt_emb,
                model.roi_heads.relation.predictor.KERN_rel_reasoning,
                model.roi_heads.relation.predictor.rel_feat_downdim_fc,
                model.roi_heads.relation.predictor.rel_feat_updim_fc,
            ]
        if cfg.MODEL.ROI_RELATION_HEAD.PREDICTOR == "IMPPredictor":
            eval_modules = [
                model.rpn,
                model.backbone,
                model.roi_heads.box,
                model.roi_heads.relation.box_feature_extractor,
                model.roi_heads.relation.union_feature_extractor,
                model.roi_heads.relation.predictor.context_layer,
            ]

        if cfg.MODEL.ROI_RELATION_HEAD.PREDICTOR == "MSDNPredictor":
            eval_modules = [
                model.rpn,
                model.backbone,
                model.roi_heads.box,
                model.roi_heads.relation.box_feature_extractor,
                model.roi_heads.relation.union_feature_extractor,
                model.roi_heads.relation.predictor.context_layer,
            ]

        if cfg.MODEL.ROI_RELATION_HEAD.PREDICTOR == "GPSNetPredictor":
            eval_modules = [
                model.rpn,
                model.backbone,
                model.roi_heads.box,
                model.roi_heads.relation.box_feature_extractor,
                model.roi_heads.relation.union_feature_extractor,
                model.roi_heads.relation.predictor.context_layer,
            ]

        if cfg.MODEL.ROI_RELATION_HEAD.RELATION_PROPOSAL_MODEL.SET_ON:
            if cfg.MODEL.ROI_RELATION_HEAD.RELATION_PROPOSAL_MODEL.METHOD == "rel_pn":
                rel_pn_module_ref.append(model.roi_heads.relation.rel_pn)
            elif cfg.MODEL.ROI_RELATION_HEAD.RELATION_PROPOSAL_MODEL.METHOD == "pre_clser":
                rel_pn_module_ref.append(
                    model.roi_heads.relation.predictor.context_layer.pre_rel_classifier
                )
    if cfg.MODEL.TRAIN_FIRST_STAGE_ONLY==True:
        eval_modules=eval_modules+tuple([model.roi_heads.relation])
    fix_eval_modules(eval_modules)# 这些模块会被设为不计算梯度
    set_train_modules(train_modules)#q为什么全0?

    if model.roi_heads.relation.rel_pn is not None:
        rel_on_module = (model.roi_heads.relation.rel_pn,)
    else:
        rel_on_module = None# this one

    logger.info("trainable models:")
    logger.info(show_params_status(model))#每部分参数

    # NOTE, we slow down the LR of the layers start with the names in slow_heads
    slow_heads = []
    if cfg.MODEL.ROI_RELATION_HEAD.PREDICTOR == "IMPPredictor":
        slow_heads = [
            "roi_heads.relation.rel_box_feature_extractor",
            "roi_heads.relation.union_feature_extractor.feature_extractor",
        ]

    elif cfg.MODEL.ROI_RELATION_HEAD.PREDICTOR == "BGNN_MODULE":
        if cfg.MODEL.ROI_RELATION_HEAD.BGNN_MODULE.RELATION_CONFIDENCE_AWARE:
            slow_heads = [
                "roi_heads.relation.predictor.context_layer.relation_conf_aware_models",
            ]        

    except_weight_decay = []
    if cfg.MODEL.ROI_RELATION_HEAD.PREDICTOR == "BGNN_MODULE":
        if (
            cfg.MODEL.ROI_RELATION_HEAD.BGNN_MODULE.RELATION_CONFIDENCE_AWARE
            and cfg.MODEL.ROI_RELATION_HEAD.BGNN_MODULE.RELNESS_MP_WEIGHTING_SCORE_RECALIBRATION_METHOD
            == "learnable_scaling"
        ):
            except_weight_decay = [
                "roi_heads.relation.predictor.context_layer.learnable_relness_score_gating_recalibration",
            ]

    # load pretrain layers to new layers
    load_mapping = {
        "roi_heads.relation.box_feature_extractor": "roi_heads.box.feature_extractor",
        "roi_heads.relation.rel_pair_box_feature_extractor": "roi_heads.box.feature_extractor",
        "roi_heads.relation.union_feature_extractor.feature_extractor": "roi_heads.box.feature_extractor",
    }

    if cfg.MODEL.ROI_RELATION_HEAD.PREDICTOR == "IMPPredictor":
        load_mapping[
            "roi_heads.relation.predictor.obj_classifier"
        ] = "roi_heads.relation.predictor.context_layer.obj_fc"
        load_mapping[
            "roi_heads.relation.predictor.rel_classifier"
        ] = "roi_heads.relation.predictor.context_layer.rel_fc"

    if cfg.MODEL.ATTRIBUTE_ON:
        load_mapping[
            "roi_heads.relation.att_feature_extractor"
        ] = "roi_heads.attribute.feature_extractor"
        load_mapping[
            "roi_heads.relation.union_feature_extractor.att_feature_extractor"
        ] = "roi_heads.attribute.feature_extractor"

    print("load model to GPU")
    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    num_batch = cfg.SOLVER.IMS_PER_BATCH #12
    optimizer = make_optimizer(
        cfg,
        model,
        logger,
        slow_heads=slow_heads,#空
        slow_ratio=2.5,
        # rl_factor=float(num_batch),
        rl_factor=1.0,
        except_weight_decay=except_weight_decay,
    )
    scheduler = make_lr_scheduler(cfg, optimizer, logger)
    debug_print(logger, "end optimizer and shcedule")
    # Initialize mixed-precision training
    use_mixed_precision = cfg.DTYPE == "float16"# false
    amp_opt_level = "O1" if use_mixed_precision else "O0"#amp用于混合精度训练
    model, optimizer = amp.initialize(model, optimizer, opt_level=amp_opt_level)

    save_to_disk = get_rank() == 0
    checkpointer = DetectronCheckpointer(
        cfg, model, optimizer, scheduler, output_dir, save_to_disk, custom_scheduler=True
    )
    # todo, unless mark as resume, otherwise load from the pretrained checkpoint
    arguments = {}
    if cfg.MODEL.PRETRAINED_DETECTOR_CKPT != "":#todo 加入对先前训练最佳值的记录
        checkpoint=checkpointer.load(
            cfg.MODEL.PRETRAINED_DETECTOR_CKPT, with_optim=False,update_schedule=False , load_mapping=load_mapping
        )
        if cfg.MODEL.PRETRAINED_DETECTOR_CKPT=='checkpoints/detection/pretrained_faster_rcnn/vg_faster_det.pth':#如果从detection模型开始训练，起始Iter应设置0
            arguments["iteration"]=0
        else:
            arguments["iteration"] =checkpoint["iteration"]
        del checkpoint
    else:
        checkpointer.load(
            cfg.MODEL.WEIGHT,

        )

        arguments["iteration"] = 0

    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    debug_print(logger, "end load checkpointer")

    if cfg.MODEL.ROI_RELATION_HEAD.RE_INITIALIZE_CLASSIFIER:
        model.roi_heads.relation.predictor.init_classifier_weight()

    # preserve a reference for logging
    rel_model_ref = model.roi_heads.relation

    debug_print(logger, "Start initializing dataset & dataloader")

    cluster_data_loader = make_data_loader(
        cfg,
        mode="train",
        is_distributed=distributed,
        start_iter=0,
        for_cluster=True
    )

    train_data_loader = make_data_loader(
        cfg,
        mode="train",
        is_distributed=distributed,
        start_iter=arguments["iteration"],
    )  # 包括dataset和dataloader类
    val_data_loaders = make_data_loader(
        cfg,
        mode="val",
        is_distributed=distributed,
    )

    debug_print(logger, "end dataloader")

    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            # this should be removed if we update BatchNorm stats
            broadcast_buffers=False,
            find_unused_parameters=True,
        )
    debug_print(logger, "end distributed")

    if cfg.SOLVER.PRE_VAL:
        logger.info("Validate before training")
        run_val(cfg, model, val_data_loaders, distributed, logger)

    pre_clser_pretrain_on = False
    if (#true
        cfg.MODEL.ROI_RELATION_HEAD.RELATION_PROPOSAL_MODEL.PRETRAIN_RELNESS_MODULE
        and cfg.MODEL.ROI_RELATION_HEAD.RELATION_PROPOSAL_MODEL.SET_ON
    ):
        if distributed:
            m2opt = model.module
        else:
            m2opt = model
        m2opt.roi_heads.relation.predictor.start_preclser_relpn_pretrain()
        logger.info("Start preclser_relpn_pretrain")
        pre_clser_pretrain_on = True

        STOP_ITER = (
            cfg.MODEL.ROI_RELATION_HEAD.RELATION_PROPOSAL_MODEL.PRETRAIN_ITER_RELNESS_MODULE
        )

    logger.info("Start training")
    meters = MetricLogger(delimiter="  ")
    max_iter = len(train_data_loader)
    start_iter = arguments["iteration"]
    start_training_time = time.time()
    end = time.time()
    model.train()
    print_first_grad = True
    if cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
        if cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
            mode = "predcls"
            recall_highest_setting = 0.29  # 暂时
        else:
            mode = "sgcls"
            recall_highest_setting = 0.10#0.15
    else:
        mode = "sgdet"
        recall_highest_setting = 0.04#0.1715  # 暂时

    if cfg.USE_CLUSTER==True:
        if os.path.isfile(cfg.OUTPUT_DIR+'/cluster_on_dataset.pkl')==False:#存放数据集Instance的feature文件
            feature=compute_features(cluster_data_loader,len(cluster_data_loader.dataset))
            synchronize()
            size_list = [torch.LongTensor([0]).to(device) for _ in range(int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1)]
            feature=all_gather(feature)
            feature=torch.cat(feature,0)
            with open(os.path.join(cfg.OUTPUT_DIR, "cluster_on_dataset.pkl"), 'wb') as f:
                pickle.dump(feature, f)
        else:
            with open(os.path.join(cfg.OUTPUT_DIR, "cluster_on_dataset.pkl"), 'rb') as f:
                feature=pickle.load(f)
        #clustering algorithm tomerger use
        if is_main_process():
            deepcluster = clustering.__dict__['Kmeans'](3)
        # clustering_loss = deepcluster.faiss_cluster(feature.to('cpu').numpy())
            clustering_loss = deepcluster.sklearn_cluster(feature.to('cpu').numpy())

    for iteration, (images, targets, _) in (enumerate(train_data_loader, start_iter)):
        # torch.cuda.empty_cache()
        if any(len(target) < 1 for target in targets):
            logger.error(
                f"Iteration={iteration + 1} || Image Ids used for training {_} || targets Length={[len(target) for target in targets]}"
            )
        data_time = time.time() - end
        iteration = iteration + 1
        arguments["iteration"] = iteration

        model.train()
        fix_eval_modules(eval_modules)

        images = images.to(device)
        targets = [target.to(device) for target in targets]

        loss_dict = model(images, targets, logger=logger) #predcls:dict:4


        losses = sum(loss for loss in loss_dict.values())
        # if cfg.MODEL.ROI_RELATION_HEAD.PREDICTOR=='MotifPredictor' and cfg.MODEL.ROI_RELATION_HEAD.OBJECT_CLASSIFICATION_REFINE:
        #     losses = 2 * loss_dict['loss_rel'] + 2 * loss_dict['loss_two_stage'] + sum(
        #         loss for loss in loss_dict.values())

        # if cfg.MODEL.TWO_STAGE_ON: #只有使用2stage时候才考虑loss为0的问题
        #     if losses==0 or ('loss_two_stage' not in loss_dict.keys()):
        #         print('counter nan: pass this iter\n')
        #         print('print loss_dict: .{}'.format(loss_dict))

            # num_gpus=get_world_size()
            # get_rank() == 0
            # optimizer.zero_grad()
            # continue
        # losses=loss_dict['loss_two_stage']+2*loss_dict['loss_two_stage']+0.5*sum(loss for loss in loss_dict.values())
        # reduce losses over all GPUs for logging purposes

        loss_dict_reduced = reduce_loss_dict(loss_dict)#
        # print('rank:{}'.format(get_rank()))
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        #for motif

        meters.update(loss=losses_reduced, **loss_dict_reduced)
        optimizer.zero_grad()
        # Note: If mixed precision is not used, this ends up doing nothing
        # Otherwise apply loss scaling for mixed-precision recipe
        # try:

        with amp.scale_loss(losses, optimizer) as scaled_losses:
            scaled_losses.backward()
        if not SHOW_COMP_GRAPH and get_rank() == 0:
            try:
                g = vis_graph.visual_computation_graph(
                    losses, model.named_parameters(), cfg.OUTPUT_DIR, "total_loss-graph"
                )
                g.render()
                for name, ls in loss_dict_reduced.items():
                    g = vis_graph.visual_computation_graph(
                        losses, model.named_parameters(), cfg.OUTPUT_DIR, f"{name}-graph"
                    )
                    g.render()
            except:
                logger.info("print computational graph failed")

            SHOW_COMP_GRAPH = True

        # add clip_grad_norm from MOTIFS, tracking gradient, used for debug
        verbose = (
            iteration % cfg.SOLVER.PRINT_GRAD_FREQ
        ) == 0 or print_first_grad  # print grad or not
        print_first_grad = False
        clip_grad_norm(
            [(n, p) for n, p in model.named_parameters() if p.requires_grad],
            max_norm=cfg.SOLVER.GRAD_NORM_CLIP,#5.0
            logger=logger,
            verbose=verbose,
            clip=True,
        )

        optimizer.step()

        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time, data=data_time)

        eta_seconds = meters.time.global_avg * (max_iter - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
        elapsed_time = str(datetime.timedelta(seconds=int(end - start_training_time)))

        if (
            iteration
            in [
                cfg.MODEL.ROI_RELATION_HEAD.RELATION_PROPOSAL_MODEL.FIX_MODEL_AT_ITER,# 1 in 3000
            ]
            and rel_on_module is not None# None
        ):
            logger.info("fix the rel pn module")
            fix_eval_modules(rel_pn_module_ref)

        if pre_clser_pretrain_on:
            if iteration == STOP_ITER:
                logger.info("pre clser pretraining ended.")
                m2opt.roi_heads.relation.predictor.end_preclser_relpn_pretrain()
                pre_clser_pretrain_on = False

        if iteration % 30 == 0:
            logger.log(TFBoardHandler_LEVEL, (meters.meters, iteration))

            logger.log(
                TFBoardHandler_LEVEL,
                ({"curr_lr": float(optimizer.param_groups[0]["lr"])}, iteration),#只是显示了第一层的learning rate
            )
            # save_buffer(output_dir)

        if iteration % 50 == 0 or iteration == max_iter:
            logger.info(
                meters.delimiter.join(
                    [
                        "\ninstance name: {instance_name}\n" "elapsed time: {elapsed_time}\n",
                        "eta: {eta}\n",
                        "iter: {iter}/{max_iter}\n",
                        "{meters}",
                        "lr: {lr:.6f}\n",
                        "max mem: {memory:.0f}\n",
                    ]
                ).format(
                    instance_name=cfg.OUTPUT_DIR[len("checkpoints/") :],
                    eta=eta_string,
                    elapsed_time=elapsed_time,
                    iter=iteration,
                    meters=str(meters),
                    lr=optimizer.param_groups[0]["lr"],
                    max_iter=max_iter,
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                )
            )
            if pre_clser_pretrain_on:
                logger.info("relness module pretraining..")
        arguments["optimizer"] =optimizer
        arguments["scheduler"] = scheduler

        val_result_value = None  # used for scheduler updating
        if cfg.SOLVER.TO_VAL and iteration % cfg.SOLVER.VAL_PERIOD == 0:
            logger.info("Start validating")
            val_result = run_val(cfg, model, val_data_loaders, distributed, logger)#->inference->vg_evaluation->evaluate_relation_of_one_image->calculate_recall
            val_result_value = val_result[1]
            if get_rank() == 0:
                for each_ds_eval in val_result[0]:
                    for each_evalator_res in each_ds_eval[1]:
                        logger.log(TFBoardHandler_LEVEL, (each_evalator_res, iteration))
        # scheduler should be called after optimizer.step() in pytorch>=1.1.0
        # https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate
        # torch.cuda.empty_cache()
        if iteration % checkpoint_period == 0 and val_result_value>recall_highest_setting:#checkpoint_period=2000
            checkpointer.save("model_{:07d}".format(iteration), **arguments)
            print("model_{:07d} beyond recall@100: recall_highest_setting".format(iteration,recall_highest_setting))
            recall_highest_setting=val_result_value
        if iteration == max_iter and val_result_value>recall_highest_setting:
            checkpointer.save("model_final", **arguments)
        restart_scheduler=True
        if cfg.SOLVER.SCHEDULE.TYPE == "WarmupReduceLROnPlateau":# default no
            if restart_scheduler:
                scheduler.step(val_result_value, epoch=iteration-start_iter)
            else:
                scheduler.step(val_result_value, epoch=iteration)
            if scheduler.stage_count >= cfg.SOLVER.SCHEDULE.MAX_DECAY_STEP:
                logger.info("Trigger MAX_DECAY_STEP at iteration {}.".format(iteration))
                break
        else:
            scheduler.step(epoch=iteration)


    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (max_iter)
        )
    )
    return model


def fix_eval_modules(eval_modules):
    for module in eval_modules:
        if module is None:
            continue

        for _, param in module.named_parameters():
            param.requires_grad = False
        # DO NOT use module.eval(), otherwise the module will be in the test mode, i.e., all self.training condition is set to False


def set_train_modules(modules):
    for module in modules:
        for _, param in module.named_parameters():
            param.requires_grad = True


def run_val(cfg, model, val_data_loaders, distributed, logger):

    # mode
    if cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
        if cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
            mode = "predcls"
        else:
            mode = "sgcls"
    else:
        mode = "sgdet"
    if distributed:
        model = model.module

    iou_types = ()
    # if mode=="sgdet":
    #     iou_types = ("bbox",)
    if cfg.MODEL.MASK_ON:
        iou_types = iou_types + ("segm",)
    if cfg.MODEL.KEYPOINT_ON:
        iou_types = iou_types + ("keypoints",)
    if cfg.MODEL.RELATION_ON:
        iou_types = iou_types + ("relations",)
    if cfg.MODEL.ATTRIBUTE_ON:
        iou_types = iou_types + ("attributes",)

    dataset_names = cfg.DATASETS.VAL#'VG_stanford_filtered_with_attribute_val'
    val_result = []
    for dataset_name, val_data_loader in zip(dataset_names, val_data_loaders):
        # torch.cuda.empty_cache()
        dataset_result = inference(
            cfg,
            model,
            val_data_loader,
            dataset_name=dataset_name,
            iou_types=iou_types,
            box_only=False if cfg.MODEL.RETINANET_ON else cfg.MODEL.RPN_ONLY,
            device=cfg.MODEL.DEVICE,
            expected_results=cfg.TEST.EXPECTED_RESULTS,
            expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
            output_folder=None,
            logger=logger,
        )
        synchronize()
        val_result.append(dataset_result)

    val_values = []
    for each in val_result:
        if isinstance(each, tuple):
            val_values.append(each[0])
    # support for multi gpu distributed testing
    # send evaluation results to each process
    gathered_result = all_gather(torch.tensor(val_values).cpu())
    gathered_result = [t.view(-1) for t in gathered_result]
    gathered_result = torch.cat(gathered_result, dim=-1).view(-1)
    valid_result = gathered_result[gathered_result >= 0]
    val_result_val = float(valid_result.mean())

    del gathered_result, valid_result
    return val_result, val_result_val


def run_test(cfg, model, distributed, logger):
    if distributed:
        model = model.module
    iou_types = ("bbox",)
    if cfg.MODEL.MASK_ON:
        iou_types = iou_types + ("segm",)
    if cfg.MODEL.KEYPOINT_ON:
        iou_types = iou_types + ("keypoints",)
    if cfg.MODEL.RELATION_ON:
        iou_types = iou_types + ("relations",)
    if cfg.MODEL.ATTRIBUTE_ON:
        iou_types = iou_types + ("attributes",)
    output_folders = [None] * len(cfg.DATASETS.TEST)
    dataset_names = cfg.DATASETS.TEST
    if cfg.OUTPUT_DIR:
        for idx, dataset_name in enumerate(dataset_names):
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
            mkdir(output_folder)
            output_folders[idx] = output_folder
    data_loaders_val = make_data_loader(cfg, mode="test", is_distributed=distributed)
    for output_folder, dataset_name, data_loader_val in zip(
        output_folders, dataset_names, data_loaders_val
    ):
        inference(
            cfg,
            model,
            data_loader_val,
            dataset_name=dataset_name,
            iou_types=iou_types,
            box_only=False if cfg.MODEL.RETINANET_ON else cfg.MODEL.RPN_ONLY,
            device=cfg.MODEL.DEVICE,
            expected_results=cfg.TEST.EXPECTED_RESULTS,
            expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
            output_folder=output_folder,
            logger=logger,
        )
        synchronize()


def main():

    torch.multiprocessing.set_start_method('forkserver')
    parser = argparse.ArgumentParser(description="PyTorch Relation Detection Training")
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--skip-test",
        default='True',
        dest="skip_test",
        help="Do not test the final model",
        action="store_true",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument('--clustering', type=str, choices=['Kmeans', 'PIC'], default='Kmeans')

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank) #default 0
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    # mode
    if cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
        if cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:#这里表示能用object label就是pre..这个任务
            mode = "predcls"
        else:
            mode = "sgcls"
    else:
        mode = "sgdet"

    now = datetime.datetime.now()
    time_str = now.strftime("%Y-%m-%d_%H")
   
    cfg.OUTPUT_DIR = os.path.join(cfg.OUTPUT_DIR,
    f"{mode}-{cfg.MODEL.ROI_RELATION_HEAD.PREDICTOR}",cfg.EXPERIMENT_NAME)
    # cfg.OUTPUT_DIR = os.path.join(
    #     cfg.OUTPUT_DIR,
    #     f"{mode}-{cfg.MODEL.ROI_RELATION_HEAD.PREDICTOR}",
    #     f"({time_str}){cfg.EXPERIMENT_NAME}"
    #     + f"{'(resampling)' if cfg.MODEL.ROI_RELATION_HEAD.DATA_RESAMPLING else ''}"
    #     + f"{'(debug)' if cfg.DEBUG else ''}",
    # )

    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir:
        mkdir(output_dir)

    logger = setup_logger("pysgg", output_dir, get_rank())#从这里进入声明tfboard
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)  #logger后面打印所有arg的参数
    # if cfg.DEBUG:
    #     logger.info("Collecting env info (might take some time)")
    #     logger.info("\n" + collect_env_info())

    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    output_config_path = os.path.join(cfg.OUTPUT_DIR, "config.yml")
    logger.info("Saving config into: {}".format(output_config_path))
    # save overloaded model config in the output directory
    save_config(cfg, output_config_path)

    model = train(cfg, args.local_rank, args.distributed, logger)

    if not args.skip_test:
        run_test(cfg, model, args.distributed, logger)


if __name__ == "__main__":
    # os.environ["OMP_NUM_THREADS"] = "12"


    main()
