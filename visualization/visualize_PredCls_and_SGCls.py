import torch
import json
import h5py
from PIL import Image, ImageDraw , ImageFont

import os

import matplotlib.pyplot as plt

image_file = json.load(open(os.path.join('/root/PySGG-cxg/datasets/vg/image_data.json')))
vocab_file = json.load(open('datasets/vg/VG-SGG-dicts-with-attri.json'))
data_file = h5py.File('datasets/vg/VG-SGG-with-attri.h5', 'r')
# remove invalid image
corrupted_ims = [1592, 1722, 4616, 4617]
tmp = []
for item in image_file:
    if int(item['image_id']) not in corrupted_ims:
        tmp.append(item)
image_file = tmp

# load detected results
detected_origin_path = '/root/PySGG-cxg/checkpoints/predcls-BGNNPredictor/cxg2/inference/VG_stanford_filtered_with_attribute_test/'
detected_origin_result = torch.load(detected_origin_path + 'eval_results.pytorch')#里面按照gt pd划分，里面由boxlist组成
detected_info = json.load(open(detected_origin_path + 'visual_info.json'))#存有img路径，还有gt pd的detecrtion结果
save_img_path=detected_origin_path+'result_img/'
# get image info by index
def get_info_by_idx(idx, det_input, thres=0.5):
    groundtruth = det_input['groundtruths'][idx]
    prediction = det_input['predictions'][idx]
    # image path
    img_path = detected_info[idx]['img_file']
    # boxes
    boxes = groundtruth.bbox
    # object labels
    idx2label = vocab_file['idx_to_label']
    labels = ['{}-{}'.format(idx,idx2label[str(i)]) for idx, i in enumerate(groundtruth.get_field('labels').tolist())]
    pred_labels = ['{}-{}'.format(idx,idx2label[str(i)]) for idx, i in enumerate(prediction.get_field('pred_labels').tolist())]
    # groundtruth relation triplet
    idx2pred = vocab_file['idx_to_predicate']#'1': 'above', '2': 'across'..
    gt_rels = groundtruth.get_field('relation_tuple').tolist()
    gt_rels = [(labels[i[0]], idx2pred[str(i[2])], labels[i[1]]) for i in gt_rels]
    # prediction relation triplet
    pred_rel_pair = prediction.get_field('rel_pair_idxs').tolist()
    pred_rel_scores = prediction.get_field('pred_rel_scores')
    pred_rel_scores[:,0] = 0
    pred_rel_score, pred_rel_label = pred_rel_scores.max(-1)
    #mask = pred_rel_score > thres
    #pred_rel_score = pred_rel_score[mask]
    #pred_rel_label = pred_rel_label[mask]
    pred_rels = [(pred_labels[i[0]], idx2pred[str(j)], pred_labels[i[1]]) for i, j in zip(pred_rel_pair, pred_rel_label.tolist())]
    return img_path, boxes, labels, gt_rels, pred_rels, pred_rel_score, pred_rel_label


def draw_single_box(pic, box,obj_box=None, pred_rels=None,  color='red',draw_info=None):
    draw = ImageDraw.Draw(pic)
    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])

    if draw_info!=None:
        draw.rectangle(((x1, y1), (x2, y2)), outline=color)
        draw.rectangle(((x1, y1), (x1 + 50, y1 + 10)), fill=color)#object的框
        info = draw_info
        draw.text((x1, y1), info)
    '''标注predict relation'''
    if obj_box!=None:

        objx1, objy1, objx2, objy2 = int(obj_box[0]), int(obj_box[1]), int(obj_box[2]), int(obj_box[3])
        # draw.rectangle((((x1+objx1)/2, (y1+objy1)/2), ((x1+objx1)/2 + 50, (y1+objy1)/2 + 10)), fill=color)  # object的框
        rel_info = pred_rels[1]
        draw.text(((x1+objx1)/2, (y1+objy1)/2), rel_info)
        shape=[(x1,y1),(objx1, objy1)]
        draw.line(shape, fill="blue", width=0)

def print_list(name, input_list):
    for i, item in enumerate(input_list):
        print(name + ' ' + str(i) + ': ' + str(item))


def draw_image(img_path, boxes, labels, gt_rels, pred_rels, pred_rel_score, pred_rel_label,idx, print_img=True):
    pic = Image.open(img_path)

    num_obj = boxes.shape[0]
    only_display_box=False

    for i in range(num_obj):
        info = labels[i]
        draw_single_box(pic, boxes[i],  draw_info=info)
    if only_display_box==False:

        # for i in range(len(pred_rels)):
        #     sub = pred_rels[i][0]
        #     obj = pred_rels[i][2]
        #     sub_idx = int(sub.split('-')[0])
        #     obj_idx=int(obj.split('-')[0])
        #     draw_single_box(pic, boxes[sub_idx],boxes[obj_idx], pred_rels[i],color='blue')

        for i in range(len(gt_rels)):
            sub = gt_rels[i][0]
            obj = gt_rels[i][2]
            sub_idx = int(sub.split('-')[0])
            obj_idx=int(obj.split('-')[0])
            draw_single_box(pic, boxes[sub_idx],boxes[obj_idx], gt_rels[i],color='black')
    if print_img:
        # display(pic)
        pic.show()
        pic.save(save_img_path+'{}.png'.format(idx))

        # plt.imshow(pic)
    if print_img:
        print('*' * 50)
        print_list('gt_boxes', labels)
        print('*' * 50)
        print_list('gt_rels', gt_rels)
        print('*' * 50)
    print_list('pred_rels', pred_rels[:20])
    print('*' * 50)

    return None


def show_selected(idx_list):
    for select_idx in idx_list:
        print(select_idx)
        draw_image(*get_info_by_idx(select_idx, detected_origin_result))


def show_all(start_idx, length):
    for cand_idx in range(start_idx, start_idx + length):
        print(cand_idx)
        draw_image(*get_info_by_idx(cand_idx, detected_origin_result),cand_idx)

show_all(start_idx=10, length=20)
#show_selected([119, 967, 713, 5224, 19681, 25371])
