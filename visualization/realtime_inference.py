# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import argparse
import cv2
import json

from pysgg.config import cfg
from demo.predictor import COCODemo

import time
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip


def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Webcam Demo")
    parser.add_argument(
        "--config-file",
        default="../configs/caffe2/e2e_mask_rcnn_R_50_FPN_1x_caffe2.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.7,
        help="Minimum score for the prediction to be shown",
    )
    parser.add_argument(
        "--min-image-size",
        type=int,
        default=224,
        help="Smallest size of the image to feed to the model. "
            "Model was trained with 800, which gives best results",
    )
    parser.add_argument(
        "--show-mask-heatmaps",
        dest="show_mask_heatmaps",
        help="Show a heatmap probability for the top masks-per-dim masks",
        action="store_true",
    )
    parser.add_argument(
        "--masks-per-dim",
        type=int,
        default=2,
        help="Number of heatmaps per dimension to show",
    )
    parser.add_argument(
        "opts",
        help="Modify model config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    # load config from file and command-line arguments
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    # prepare object that handles inference plus adds predictions on top of image
    coco_demo = COCODemo(
        cfg,
        confidence_threshold=args.confidence_threshold,
        show_mask_heatmaps=args.show_mask_heatmaps,
        masks_per_dim=args.masks_per_dim,
        min_image_size=args.min_image_size,
    )
    ffmpeg_extract_subclip('checkpoints/predcls-BGNNPredictor/cxg2/inference/Los.mp4', 45, 50, targetname="test.mp4")
    cam = cv2.VideoCapture('test.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter('output_witnrelation.mp4', fourcc, 20.0, (int(cam.get(3)), int(cam.get(4))))

    while (cam.isOpened()):
        start_time = time.time()
        ret_val, img = cam.read()
        if ret_val==True:
            composite = coco_demo.run_on_opencv_image(img)
            print("Time: {:.2f} s / img".format(time.time() - start_time))
            # cv2.imshow("COCO detections", composite)
            if cv2.waitKey(1) == 27:
                break  # esc to quit
            out.write(composite)
            # cv2.imshow('frame', composite)
            c = cv2.waitKey(1)
            if c & 0xFF == ord('q'):
                break
        else:
            break


    out.release()
    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
