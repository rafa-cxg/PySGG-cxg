# Biasing like human: a cognitive bias framework for scene graph generation

[![Python](https://img.shields.io/badge/python-3.7-blue.svg)](https://www.python.org/)
![PyTorch](https://img.shields.io/badge/pytorch-1.4.0-%237732a8)

Our paper [Biasing like human: a cognitive bias framework for scene graph generation] official code
<!-- (https://arxiv.org/abs/2104.00308) -->
We also announce a new scene graph branchmark as replacement of scene-graph-brenchmark and PySGG.
Except all existing feature of abrove, there are extra following updating:
- Add more state of art methods: Dual-Transformer, SHA, C-bias
- Allowing batch size more than one in test/val phrase (fixed from PySGG).
- fix truncate image error problem.
- Allow loading different size weights with same name.
- Support multi-gpu on SHA model 
## Installation

Check [INSTALL.md](INSTALL.md) for installation instructions.

## Dataset

Check [DATASET.md](DATASET.md) for instructions of dataset preprocessing.

## Training **(IMPORTANT)**

### Prepare Faster-RCNN Detector
- You can download the pretrained Faster R-CNN we used in the paper: 
  - [VG](https://shanghaitecheducn-my.sharepoint.com/:u:/g/personal/lirj2_shanghaitech_edu_cn/EQIy64T-EK9Er9y8kVCDaukB79gJwfSsEIbey9g0Xag6lg?e=wkKHJs), 
- put the checkpoint into the folder:
```
mkdir -p checkpoints/detection/pretrained_faster_rcnn/
# for VG
mv /path/vg_faster_det.pth checkpoints/detection/pretrained_faster_rcnn/
```

Then, you need to modify the pretrained weight parameter `MODEL.PRETRAINED_DETECTOR_CKPT` in configs yaml `configs/bgnn.yaml` to the path of corresponding pretrained rcnn weight to make sure you load the detection weight parameter correctly.



### C-bias framework
You can follow the following instructions to train your own, which takes 2 GPUs for traing. The results should be very close to the reported results given in paper.
#### C-bias framework auguments:
3 paradiagms are enabled following 3 commands:
```
MODEL.TWO_STAGE_ON True #For EEM
MODEL.ROI_RELATION_HEAD.VISUAL_LANGUAGE_MERGER_EDGE True #For LMM
MODEL.ROI_RELATION_HEAD.VISUAL_LANGUAGE_MERGER_OBJ True  #For SEM
```
you can copy the following command to train
#### Scripts
For baseline bgnn, we use configration file [configs/bgnn.yaml](configs/bgnn.yaml) provided by author:
```
gpu_num=2 && python -m torch.distributed.launch --master_port 10028 --nproc_per_node=$gpu_num \
       tools/relation_train_net.py \
       --config-file "configs/bgnn.yaml" \
        DEBUG False \
        EXPERIMENT_NAME "human_bgnn" \
        MODEL.ROI_RELATION_HEAD.PREDICTOR BGNNPredictor \
        MODEL.ROI_RELATION_HEAD.USE_GT_BOX False \
        MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False \
        SOLVER.IMS_PER_BATCH $[10*$gpu_num] \
        TEST.IMS_PER_BATCH $[$gpu_num] \
        SOLVER.VAL_PERIOD 500 \
        SOLVER.CHECKPOINT_PERIOD 500\
        MODEL.PRETRAINED_DETECTOR_CKPT "checkpoints/detection/pretrained_faster_rcnn/vg_faster_det.pth"\
        SOLVER.BASE_LR 0.006 \
        DATALOADER.NUM_WORKERS 0 \
        MODEL.TWO_STAGE_ON True \
        MODEL.TWO_STAGE_HEAD.LOSS_TYPE 'cos_loss' \
        MODEL.ROI_RELATION_HEAD.VISUAL_LANGUAGE_MERGER_EDGE True \
        MODEL.ROI_RELATION_HEAD.VISUAL_LANGUAGE_MERGER_OBJ True \
        
        
```
For baseline MOTIFs, IMP, G-RCNN, Transformer..., ypu just need to change `MODEL.ROI_RELATION_HEAD.PREDICTOR` to one of `MotifPredictor`, `IMPPredictor`, `AGRCNNPredictor`,`TransformerPredictor`:
```
gpu_num=2 && python -m torch.distributed.launch --master_port 10028 --nproc_per_node=$gpu_num \
       tools/relation_train_net.py \
       --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" \
        DEBUG False \
        EXPERIMENT_NAME "human_motif" \
        MODEL.ROI_RELATION_HEAD.PREDICTOR MotifPredictor \
        MODEL.ROI_RELATION_HEAD.USE_GT_BOX False \
        MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False \
        SOLVER.IMS_PER_BATCH $[6*$gpu_num] \
        TEST.IMS_PER_BATCH $[$gpu_num] \
        SOLVER.VAL_PERIOD 500 \
        SOLVER.CHECKPOINT_PERIOD 500\
        MODEL.PRETRAINED_DETECTOR_CKPT "checkpoints/detection/pretrained_faster_rcnn/vg_faster_det.pth"\
        SOLVER.BASE_LR 0.02 \
        DATALOADER.NUM_WORKERS 0 \
        MODEL.TWO_STAGE_ON True \
        MODEL.TWO_STAGE_HEAD.LOSS_TYPE 'cos_loss' \
        MODEL.ROI_RELATION_HEAD.VISUAL_LANGUAGE_MERGER_EDGE True \
        MODEL.ROI_RELATION_HEAD.VISUAL_LANGUAGE_MERGER_OBJ True \
        
        
```
For baseline Unbiasd, all you need to do is set those :
```
MODEL.ROI_RELATION_HEAD.PREDICTOR CausalAnalysisPredictor \
MODEL.ROI_RELATION_HEAD.CAUSAL.AUXILIARY_LOSS True \
MODEL.ROI_RELATION_HEAD.CAUSAL.CONTEXT_LAYER bgnn #or motifs \
MODEL.ROI_RELATION_HEAD.CAUSAL.EFFECT_ANALYSIS true \
MODEL.ROI_RELATION_HEAD.CAUSAL.EFFECT_TYPE TDE \
``` 


## Test
You can directly check performance of model from the checkpoint. By replacing the parameter of `MODEL.WEIGHT` to the trained model weight and selected dataset name in `DATASETS.TEST`, you can directly eval the model on validation or test set.
```
archive_dir="checkpoints/predcls-BGNNPredictor/xxx/xxx"

python -m torch.distributed.launch --master_port 10029 --nproc_per_node=$gpu_num  \
  tools/relation_test_net.py \
  --config-file "$archive_dir/config.yml"\
    TEST.IMS_PER_BATCH $[$gpu_num] \
   MODEL.WEIGHT  "$archive_dir/model_xxx.pth"\
   MODEL.ROI_RELATION_HEAD.EVALUATE_REL_PROPOSAL False \
   DATASETS.TEST "('VG_stanford_filtered_with_attribute_eval', )"

```
### visiualization
Firstly, you need prepare `eval_results.pytorch` and `visual_info.json` generated by `relation_test_net.py`. Then, change `detected_origin_path` your results path. You can get image with predicted box and predicted relation in output stream by:
```
python visualization/visualize_PredCls_and_SGCls.py --detected_origin_path <your path/> --start_idx 0 --end_idx 100 --draw_pred_box True --draw_gt_box False 
```

## Citations

If you find this project helps your research, please kindly consider citing our papers in your publications.

```

```


## Acknowledgment
This repository is developed on top of the scene graph benchmark Toolkit [PySGG](https://github.com/SHTUPLUS/PySGG)
