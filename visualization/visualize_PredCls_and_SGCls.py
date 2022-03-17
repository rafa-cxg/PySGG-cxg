import torch
import json
import h5py
from PIL import Image, ImageDraw , ImageFont
import argparse

import os

import matplotlib.pyplot as plt

image_file = json.load(open(os.path.join('datasets/vg/image_data.json')))
vocab_file = json.load(open('datasets/vg/VG-SGG-dicts-with-attri.json'))
data_file = h5py.File('datasets/vg/VG-SGG-with-attri.h5', 'r')
# remove invalid image
corrupted_ims = [1592, 1722, 4616, 4617]
tmp = []
for item in image_file:
    if int(item['image_id']) not in corrupted_ims:
        tmp.append(item)
image_file = tmp

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
    pred_labels = ['{}-{}'.format(idx, idx2label[str(i)]) for idx, i in enumerate(prediction.get_field('pred_labels').tolist())]
    # groundtruth relation triplet
    idx2pred = vocab_file['idx_to_predicate']#'1': 'above', '2': 'across'..
    gt_rels = groundtruth.get_field('relation_tuple').tolist()
    gt_rels = [(labels[i[0]], idx2pred[str(i[2])], labels[i[1]]) for i in gt_rels]
    # prediction relation triplet
    pred_rel_pair = prediction.get_field('rel_pair_idxs').tolist()
    pred_rel_scores = prediction.get_field('pred_rel_scores')
    pred_rel_scores[:,0] = 0
    pred_rel_score, pred_rel_label = pred_rel_scores.max(-1) #[n,51]
    #mask = pred_rel_score > thres
    #pred_rel_score = pred_rel_score[mask]
    #pred_rel_label = pred_rel_label[mask]
    pred_rels = [(pred_labels[i[0]], idx2pred[str(j)], pred_labels[i[1]]) for i, j in zip(pred_rel_pair, pred_rel_label.tolist())]



    return img_path, boxes,labels, gt_rels, pred_rels, pred_rel_score, pred_rel_label
def draw_pred_box(idx, det_input,color='blue',print_img=True):
    img_path = detected_info[idx]['img_file']
    pic = Image.open(img_path)
    # pic = Image.open(save_img_path+'{}.png'.format(idx))
    prediction = det_input['predictions'][idx]
    pred_rel_pair = prediction.get_field('rel_pair_idxs').tolist()
    # get unique pred box related to top 20 relation
    list_pred_box_idx= list(set([item for subset in pred_rel_pair[:20] for item in subset]))
    idx2label = vocab_file['idx_to_label']
    pred_labels = ['{}-{}'.format(idx, idx2label[str(i)]) for idx, i in enumerate(prediction.get_field('pred_labels')[list_pred_box_idx].tolist())]

    pred_boxes = (prediction.bbox)[list_pred_box_idx]
    draw = ImageDraw.Draw(pic)
    for i in range(pred_boxes.shape[0]):
        x1, y1, x2, y2 = int(pred_boxes[i][0]), int(pred_boxes[i][1]), int(pred_boxes[i][2]), int(pred_boxes[i][3])
        draw.rectangle(((x1, y1), (x2, y2)), outline=color,width=2)
        # draw.rectangle(((x1, y1), (x1 + 50, y1 + 10)), fill=color)#object的框
        draw.text((x1, y1), pred_labels[i])
    if print_img:
        pic.save(save_img_path+'{}.png'.format(idx))
        pic.close()
    if args.draw_gt_box ==False:  #prevent duplicate
        print_list('pred_rels', pred_labels[:20])
        print('*' * 50)
def draw_single_box(pic, box,obj_box=None, pred_rels=None,  color='red',draw_info=None):
    draw = ImageDraw.Draw(pic)
    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
 
    if draw_info!=None:
        draw.rectangle(((x1, y1), (x2, y2)), outline=color,width=2)
        # draw.rectangle(((x1, y1), (x1 + 50, y1 + 10)), fill=color)#object的框
        info = draw_info
        # font=ImageFont.truetype('/usr/share/font/truetype/freefont/simsun.ttc',size=6)
        draw.text((x1, y1), info)
    '''标注predict relation'''
    if obj_box!=None:

        objx1, objy1, objx2, objy2 = int(obj_box[0]), int(obj_box[1]), int(obj_box[2]), int(obj_box[3])
        # draw.rectangle((((x1+objx1)/2, (y1+objy1)/2), ((x1+objx1)/2 + 50, (y1+objy1)/2 + 10)), fill=color)  # object的框
        # find predbox occur in relation
        rel_info = pred_rels[1]
        draw.text(((x1+objx1)/2, (y1+objy1)/2), rel_info)
        shape=[(x1,y1),(objx1, objy1)]
        draw.line(shape, fill="blue", width=0)

def print_list(name, input_list):
    for i, item in enumerate(input_list):
        print(name + ' ' + str(i) + ': ' + str(item))


def draw_image(img_path, boxes ,labels, gt_rels, pred_rels, pred_rel_score, pred_rel_label,idx, print_relation=True):
    draw_all_gt_box=False
    pic = Image.open(img_path)

    num_obj = boxes.shape[0]
    only_display_box=True#    display_predicted_box=True
    # if show all gt box
    if draw_all_gt_box:
        for i in range(num_obj):
            info = labels[i]
            draw_single_box(pic, boxes[i],  draw_info=info)

    else:
        #gt box idx that is related to gt relation
        gt_box_relative=list(set([int(list(i)[0].split('-')[0]) for i in gt_rels]+[int(list(i)[2].split('-')[0]) for i in gt_rels]))
        num_obj = boxes.shape[0]
        if print_relation:
            for i in gt_box_relative:
                info = labels[i]
                draw_single_box(pic, boxes[i],  draw_info=info)
        else: pass
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

    # display(pic)
    pic.show()
    if os.path.isdir(detected_origin_path) and os.path.exists(save_img_path)==False:
        os.mkdir(save_img_path)
    pic.save(save_img_path+'{}.png'.format(idx))
    pic.close()
    # plt.imshow(pic)

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


def show_all(start_idx, length,args):
    for cand_idx in range(start_idx, start_idx + length):
        print('image',cand_idx)
        if args.draw_gt_box:
            draw_image(*get_info_by_idx(cand_idx, detected_origin_result),cand_idx)
        if args.draw_pred_box:
            draw_pred_box(cand_idx, detected_origin_result)
        else:
            draw_image(*get_info_by_idx(cand_idx, detected_origin_result), cand_idx,print_relation=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch Relation Detection Training")
    parser.add_argument('--detected_origin_path', type=str)
    parser.add_argument('--start_idx', type=int,default=0)
    parser.add_argument('--end_idx', type=int, default=100)
    parser.add_argument('--draw_pred_box', type=bool, default=False)
    parser.add_argument('--draw_gt_box', type=bool, default=False)

    args = parser.parse_args()
    # load detected results
    detected_origin_path= args.detected_origin_path
    start_idx= args.start_idx;end_idx= args.end_idx
    
    # detected_origin_path = 'checkpoints/bgnn/inference/VG_stanford_filtered_with_attribute_test/'  # 'human_bgnn_visual/inference/VG_stanford_filtered_with_attribute_test/'#'checkpoints/bgnn/inference/VG_stanford_filtered_with_attribute_test/'
    detected_origin_result = torch.load(detected_origin_path + 'eval_results.pytorch')  # 里面按照gt pd划分，里面由boxlist组成
    detected_info = json.load(open(detected_origin_path + 'visual_info.json'))  # 存有img路径，还有gt pd的detecrtion结果
    save_img_path = detected_origin_path + 'result_img/'

    show_all(start_idx=start_idx, length=(end_idx-start_idx),args=args)
    #show_selected([119, 967, 713, 5224, 19681, 25371])
