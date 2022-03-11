import json
import logging
import os
from time import time


def load_image_filenames(img_dir, image_file, check_img_file):
    """
    Loads the image filenames from visual genome from the JSON file that contains them.
    This matches the preprocessing in scene-graph-TF-release/data_tools/vg_to_imdb.py.
    Parameters:
        image_file: JSON file. Elements contain the param "image_id".
        img_dir: directory where the VisualGenome images are located
    Return:
        List of filenames corresponding to the good images
    """
    with open(image_file, 'r') as f:
        im_data = json.load(f)
        print('load json ok')


    corrupted_ims = ['1592.jpg', '1722.jpg', '4616.jpg', '4617.jpg']
    fns = []
    img_info = []
    for i, img in enumerate(im_data):

        basename = '{}.jpg'.format(img['image_id'])
        # print(img['image_id'])

        if basename in corrupted_ims:
            continue

        filename = os.path.join(img_dir, basename)
        s=time()
        if os.path.exists(filename) or not check_img_file:
            e=time()
            total=e-s
            print('时间',total)
            fns.append(filename)
            img_info.append(img)
    assert len(fns) == 108073
    assert len(img_info) == 108073
    return fns, img_info
def main():
    load_image_filenames('/root/PySGG-cxg/datasets/vg/stanford_spilt/VG_100k_images','/root/PySGG-cxg/datasets/vg/image_data.json',True)

if __name__ == "__main__":
    main()