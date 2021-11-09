import h5py
from tblib import pickling_support
pickling_support.install()
from six import reraise
import os
import sys
import numpy as np
from itertools import islice
import argparse
from functools import partial
import orjson
# import tqdm
import pprint
import pdb
import time
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
# from scipy.misc import imread
from imageio import imread
import os
import path
import  copy
import pickle
import json
import numpy as np
import sys
import multiprocessing
from  multiprocessing import  Pool
from scipy.stats import wasserstein_distance
sys.path.append("../../coco/PythonAPI/")

from collections import defaultdict, Counter
import time
import itertools
import gc

import  sklearn
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import pairwise_distances
import pickle
from pathlib import Path
# from stanfordcorenlp import StanfordCoreNLP
MAX_PROCESS=50

props = {'annotators': 'pos,lemma',
         'pipelineLanguage': 'en',
         'outputFormat': 'json'}

# nlp = StanfordCoreNLP('http://localhost', port=10000, timeout=30000)


useless_prefix=['ion','fro','for','re','w','th','without','with','I','something','no','n','p','t','is','are','being','been','can','and','beside','on','ON','IN','in','by','upon','against','inside''into','at','out','from','of','has','have','had',' a ','a','an','to','with']

class ExceptionWrapper(object):

    def __init__(self, ee):
        self.ee = ee
        __, __, self.tb = sys.exc_info()

    def re_raise(self):
        # s = pickle.dumps(self.tb)
        # raise pickle.loads(s)
        s2 = pickle.dumps(self.tb, protocol=pickle.HIGHEST_PROTOCOL)
        raise  pickle.loads(s2)
        # raise self.ee.with_traceback(self.tb)
        # for Python 2 replace the previous line by:
        # raise self.ee, None, self.tb
def _any_in(source, target):
    for entry in source:
            return True
    return False


def _like_array(obj):
    return hasattr(obj, '__iter__') and hasattr(obj, '__len__')


def _get_cat_id(synset, cats):
    for idx in cats.keys():
        if cats[idx] == synset:
            return idx
class VG:
    def __init__(self, data_dir, annotation_file=None, num=-1, stats=False, align_dir=None):
        self.data_dir = data_dir
        self.num = num
        self.dataset = dict()  # create empty dict
        self.anns, self.abn_anns, self.cats, self.imgs,self.objects = dict(), dict(), dict(), dict(), dict() #self.objects is 1000 objects dict
        self.ann_lens, self.img_lens = {}, {}
        # self.img_to_anns, self.cat_to_imgs, self.cat_to_rel,self.rel_to_cat = defaultdict(list), defaultdict(list), defaultdict(lambda :[0]*1234),defaultdict(lambda :[0]*7003)  # img_to_anns:{image_id:list of [relationship_category_id]}
        self.img_to_anns, self.cat_to_imgs, self.cat_to_rel, self.rel_to_cat = defaultdict(list), defaultdict(
            list), defaultdict(lambda: [0] * 3485), defaultdict(lambda: [0] * 30149)#3982
        #defaultdict(lambda: [0] * 21834), defaultdict(lambda: [0] * 28368) it is for using 'name'
        self.align_list = dict()
        self.annotation_file = annotation_file
        if annotation_file is not None:
            print('loading annotations into memory...')
            tic = time.time()
            dataset = json.load(open(os.path.join(
                annotation_file), 'r'))
            print('Done (t={:0.2f}s'.format(time.time() - tic))
            self.dataset = dataset
            if align_dir is not None:
                if align_dir == 'val':
                    self.align_list[954] = 'pigeon.n.01'  # vg1000 val
                else:
                    align_path = os.path.join(self.data_dir, 'vg_' + align_dir + '_align.json')
                    self.align_list = json.load(open(align_path, 'r'))

            self.create_index()
            if stats:
                self.compute_cat_stats()

        del dataset, self.dataset
        gc.collect()
    def create_index(self):  # self:VG
        storage_dir = 'refine_result'
        print('creating index...')
        if self.num < 0:  # -1
            self.num = len(
                self.dataset)  # self.dataset is loaded raw anno file. 87946 ,dataset could by 'train' or 'val'
        for cnt, img in enumerate(self.dataset[:self.num], 1):  # img:'imageid' 'objects' 'imageurl' 'hight' 'weight'
            self.imgs[img['image_id']] = img
            if (self.annotation_file.find('rel') != -1):  # this is relation anno
                for ann in img['relationships']:

                    ann['image_id'] = img['image_id']
                    if(len(ann['new_pre'])!=0):
                        predicate = ann['new_pre']
                    else:
                        continue
                    if(len(ann['subject']['synsets'])==1):
                        sub = ann['subject']['name']
                    else:
                        continue
                    if (len(ann['object']['synsets']) == 1):
                        obj = ann['object']['name']
                    else:
                        continue
                    # except:

                    if((sub) not in  self.objects.keys()):
                        self.objects[str(sub) ] =  len(self.objects)
                    if ((obj) not in self.objects.keys()):
                        self.objects[ str(obj) ] = len(self.objects)
                    if 'relationship_category_id' not in ann: # add 'relationship_category_id' to ann
                        if predicate not in self.cats.values():
                            self.cats[len(self.cats)] = predicate
                            relationship_category_id = len(self.cats) - 1
                        else:
                            relationship_category_id = _get_cat_id(predicate, self.cats) # give a rel name and cats dict,return index (not include 'no relationship')



                        if(os.path.exists('predicate_stastic')):
                            self.cat_to_rel[(self.objects[str(sub)])][relationship_category_id] += 1
                            self.cat_to_rel[(self.objects[str(obj)])][relationship_category_id] += 1
                            pass
                        else:
                            self.cat_to_rel[(self.objects[str(sub) ])][relationship_category_id] += 1
                            self.cat_to_rel[(self.objects[str(obj) ])][relationship_category_id] += 1

                        if (os.path.exists('object_stastic_no_syn')):
                            self.rel_to_cat[str(self.cats[relationship_category_id])][
                                self.objects[str(sub)]] += 1
                            self.rel_to_cat[str(self.cats[relationship_category_id])][
                                self.objects[str(obj)]] += 1

                            pass
                        else:
                            self.rel_to_cat[str(self.cats[relationship_category_id])][self.objects[str(sub)]] += 1
                            self.rel_to_cat[str(self.cats[relationship_category_id])][self.objects[str(obj)]] += 1


                        ann['relationship_category_id'] = relationship_category_id
                    else:  # never happen cirsumstance
                        relationship_category_id = ann['relationship_category_id']
                        self.cats[relationship_category_id] = predicate  # like ['bed.n.01']
                    # self.cat_to_rel[str(synsets_sub[0])][relationship_category_id] += 1
                    self.cat_to_imgs[relationship_category_id].append(
                        img['image_id'])  # specific catagory's coppesponding image ids
                    self.img_to_anns[img['image_id']].append(
                        ann['relationship_id'])  # specific image's coppesponding object id
                    self.anns[ann[
                        'relationship_id']] = ann  # list of all relationship instance each one is like:{'predicate': 'ON', 'object': {'h': 290, 'object_id': 1058534, 'merged_object_ids': [5046], 'synsets': ['sidewalk.n.01'], 'w': 722, 'y': 308, 'x': 78, 'names': ['sidewalk']}, 'relationship_id': 15927, 'synsets': ['along.r.01'], 'subject': {'name': 'shade', 'h': 192, 'synsets': ['shade.n.01'], 'object_id': 5045, 'w': 274, 'y': 338, 'x': 119}, 'image_id': 1}


            else:
                for ann in img['objects']:  # ann is one image's each object
                    ann['image_id'] = img['image_id']
                    synsets = ann['synsets']
                    # TODO: a box may have >= 2 or 0 synsets
                    if len(synsets) != 1:
                        # self.show_cat_anns(img_in=img, ann_in=ann)
                        self.abn_anns[ann['object_id']] = ann
                    # only consider those objects with exactly one synset
                    else:  # never happen circumstance
                        synset = synsets[0]
                        if 'category_id' not in ann:
                            if synset not in self.cats.values():
                                self.cats[len(self.cats)] = synset
                                category_id = len(self.cats) - 1
                            else:
                                category_id = _get_cat_id(synset, self.cats)
                            ann['category_id'] = category_id
                        else:  # normal cirsumstance
                            category_id = ann['category_id']
                            self.cats[category_id] = synset  # like ['bed.n.01']
                        self.cat_to_imgs[category_id].append(
                            img['image_id'])  # specific catagory's coppesponding image ids
                        self.img_to_anns[img['image_id']].append(
                            ann['object_id'])  # specific image_id's coppesponding object id
                        self.anns[ann['object_id']] = ann
                        # self.cats[954] = 'pigeon.n.01' #vg1000 test
            if cnt % 100 == 0:
                print('{} images indexed...'.format(cnt))  # indexed mean decompose dict to corresponding list
            elif cnt == self.num:
                print('{} images indexed...'.format(cnt))
        if self.align_list:
            for a_i in self.align_list:
                self.cats[int(a_i)] = self.align_list[a_i]
            print("########### add lacking label done ##################")
        print('index created!')
        # cat_to_rel_temp=defaultdict(list)
        # for key, value in self.cat_to_rel:
        #     cat_to_rel_temp[int(self.cat_to_rel[str(key)])]=value
        cat_to_rel={}
        for key in self.cat_to_rel:
            cat_to_rel[(key)]=self.cat_to_rel[(key)]
        rel_to_cat = {}
        for key in self.rel_to_cat:
            rel_to_cat[str(key)] = self.rel_to_cat[str(key)]


        with open('predicate_stastic','wb') as handle:
            pickle.dump(cat_to_rel,handle,protocol=pickle.HIGHEST_PROTOCOL)
        with open('object_stastic_no_syn', 'wb') as handle:
            pickle.dump(rel_to_cat, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open('predicate', 'wb') as handle:
            pickle.dump(self.cats, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open('objects', 'wb') as handle:
            pickle.dump(self.objects, handle, protocol=pickle.HIGHEST_PROTOCOL)#26119
    def create_synset_index(self):  # self:VG
        print('creating index...')
        if self.num < 0:  # -1
            self.num = len(
                self.dataset)  # self.dataset is loaded raw anno file. 87946 ,dataset could by 'train' or 'val'
        for cnt, img in enumerate(self.dataset[:self.num], 1):  # img:'imageid' 'objects' 'imageurl' 'hight' 'weight'
            self.imgs[img['image_id']] = img
            if (self.annotation_file.find('rel') != -1):  # this is relation anno
                for ann in img['relationships']:
                    ann['image_id'] = img['image_id']
                    synsets_rel = ann['synsets']
                    if(len(ann['subject']['synsets'])==1):
                        synsets_sub = ann['subject']['synsets'][0]
                    if (len(ann['object']['synsets']) == 1):
                        synsets_obj = ann['object']['synsets'][0]
                    # except:
                    #     synsets_obj = ann['object']['name']

                    if len(synsets_rel) != 1:
                        # self.show_cat_anns(img_in=img, ann_in=ann)
                        self.abn_anns[ann['relationship_id']] = ann
                    # only consider those objects with exactly one synset
                    else:  # normal circumstance
                        synset_rel = synsets_rel[0]
                        if(str(synsets_sub) not in  self.objects.keys()):
                            self.objects[str(synsets_sub) ] =  len(self.objects)
                        if (str(synsets_obj) not in self.objects.keys()):
                            self.objects[ str(synsets_obj) ] = len(self.objects)
                        if 'relationship_category_id' not in ann: # add 'relationship_category_id' to ann
                            if synset_rel not in self.cats.values():
                                self.cats[len(self.cats)] = synset_rel
                                relationship_category_id = len(self.cats) - 1
                            else:
                                relationship_category_id = _get_cat_id(synset_rel, self.cats) # give a rel name and cats dict,return index



                            if(os.path.exists('relation_stastic_150')):
                                self.cat_to_rel[(self.objects[str(synsets_sub)])][relationship_category_id] += 1
                                self.cat_to_rel[(self.objects[str(synsets_obj)])][relationship_category_id] += 1
                                pass
                            else:
                                self.cat_to_rel[(self.objects[str(synsets_sub) ])][relationship_category_id] += 1
                                self.cat_to_rel[(self.objects[str(synsets_obj) ])][relationship_category_id] += 1

                            if (os.path.exists('object_stastic_150')):
                                self.rel_to_cat[str(self.cats[relationship_category_id])][
                                    self.objects[str(synsets_sub)]] += 1
                                self.rel_to_cat[str(self.cats[relationship_category_id])][
                                    self.objects[str(synsets_obj)]] += 1

                                pass
                            else:
                                self.rel_to_cat[str(self.cats[relationship_category_id])][self.objects[str(synsets_sub)]] += 1
                                self.rel_to_cat[str(self.cats[relationship_category_id])][self.objects[str(synsets_obj)]] += 1


                            ann['relationship_category_id'] = relationship_category_id
                        else:  # never happen cirsumstance
                            relationship_category_id = ann['relationship_category_id']
                            self.cats[relationship_category_id] = synset_rel  # like ['bed.n.01']
                        # self.cat_to_rel[str(synsets_sub[0])][relationship_category_id] += 1
                        self.cat_to_imgs[relationship_category_id].append(
                            img['image_id'])  # specific catagory's coppesponding image ids
                        self.img_to_anns[img['image_id']].append(
                            ann['relationship_id'])  # specific image's coppesponding object id
                        self.anns[ann[
                            'relationship_id']] = ann  # list of all relationship instance each one is like:{'predicate': 'ON', 'object': {'h': 290, 'object_id': 1058534, 'merged_object_ids': [5046], 'synsets': ['sidewalk.n.01'], 'w': 722, 'y': 308, 'x': 78, 'names': ['sidewalk']}, 'relationship_id': 15927, 'synsets': ['along.r.01'], 'subject': {'name': 'shade', 'h': 192, 'synsets': ['shade.n.01'], 'object_id': 5045, 'w': 274, 'y': 338, 'x': 119}, 'image_id': 1}


            else:
                for ann in img['objects']:  # ann is one image's each object
                    ann['image_id'] = img['image_id']
                    synsets = ann['synsets']
                    # TODO: a box may have >= 2 or 0 synsets
                    if len(synsets) != 1:
                        # self.show_cat_anns(img_in=img, ann_in=ann)
                        self.abn_anns[ann['object_id']] = ann
                    # only consider those objects with exactly one synset
                    else:  # never happen circumstance
                        synset = synsets[0]
                        if 'category_id' not in ann:
                            if synset not in self.cats.values():
                                self.cats[len(self.cats)] = synset
                                category_id = len(self.cats) - 1
                            else:
                                category_id = _get_cat_id(synset, self.cats)
                            ann['category_id'] = category_id
                        else:  # normal cirsumstance
                            category_id = ann['category_id']
                            self.cats[category_id] = synset  # like ['bed.n.01']
                        self.cat_to_imgs[category_id].append(
                            img['image_id'])  # specific catagory's coppesponding image ids
                        self.img_to_anns[img['image_id']].append(
                            ann['object_id'])  # specific image_id's coppesponding object id
                        self.anns[ann['object_id']] = ann
                        # self.cats[954] = 'pigeon.n.01' #vg1000 test
            if cnt % 100 == 0:
                print('{} images indexed...'.format(cnt))  # indexed mean decompose dict to corresponding list
            elif cnt == self.num:
                print('{} images indexed...'.format(cnt))
        if self.align_list:
            for a_i in self.align_list:
                self.cats[int(a_i)] = self.align_list[a_i]
            print("########### add lacking label done ##################")
        print('index created!')
        # cat_to_rel_temp=defaultdict(list)
        # for key, value in self.cat_to_rel:
        #     cat_to_rel_temp[int(self.cat_to_rel[str(key)])]=value
        cat_to_rel={}
        for key in self.cat_to_rel:
            cat_to_rel[(key)]=self.cat_to_rel[(key)]
        rel_to_cat = {}
        for key in self.rel_to_cat:
            rel_to_cat[str(key)] = self.rel_to_cat[str(key)]
        cat_to_rel=copy.deepcopy(self.cat_to_rel)
        if(os.path.exists('relation_stastic_150')!=True):
            with open('relation_stastic_150','wb') as handle:
                pickle.dump(cat_to_rel,handle,protocol=pickle.HIGHEST_PROTOCOL)
        if (os.path.exists('object_stastic_150') != True):
            with open('object_stastic_150', 'wb') as handle:
                pickle.dump(rel_to_cat, handle, protocol=pickle.HIGHEST_PROTOCOL)


    def get_ann_ids(self, cat_ids=[], img_ids=[]):
        cat_ids = cat_ids if _like_array(cat_ids) else [cat_ids]
        img_ids = img_ids if _like_array(img_ids) else [img_ids]

        if len(img_ids) > 0:
            lists = [self.img_to_anns[img_id] for img_id in img_ids
                     if img_id in self.img_to_anns]  # list is rel_category id corresponding to img_ids
            ids = list(itertools.chain.from_iterable(lists))
        else:
            ids = self.anns.keys()
        if len(cat_ids) > 0:
            ids = [idx for idx in ids if
                   self.anns[idx]['category_id'] in cat_ids]
        return sorted(ids)

    def get_cat_ids(self, cat_ids=[]):
        cat_ids = cat_ids if _like_array(cat_ids) else [cat_ids]

        ids = self.cats.keys()  # 1000 catgory, dict_keys type
        if len(cat_ids) > 0:
            ids = [cat_id for cat_id in cat_ids if cat_id in ids]  # just according to oder
        return sorted(ids)

    def get_img_ids(self, cat_ids=[], img_ids=[]):
        cat_ids = cat_ids if _like_array(cat_ids) else [cat_ids]
        img_ids = img_ids if _like_array(img_ids) else [img_ids]

        if len(img_ids) > 0:
            ids = set(img_ids) & set(self.imgs.keys())
        else:
            ids = set(self.imgs.keys())
        for i, cat_id in enumerate(cat_ids):
            if i == 0:
                ids_int = ids & set(self.cat_to_imgs[cat_id])
            else:
                ids_int |= ids & set(self.cat_to_imgs[cat_id])
        if len(cat_ids) > 0:
            return list(ids_int)
        else:
            return list(ids)

    def load_anns(self, ids=[]):
        if _like_array(ids):
            return [self.anns[idx] for idx in ids]
        elif type(ids) == int:
            return [self.anns[ids]]

    def load_cats(self, ids=[]):
        if _like_array(ids):
            return [self.cats[idx] for idx in ids]
        elif type(ids) == int:
            return [self.cats[ids]]

    def load_imgs(self, ids=[]):
        if _like_array(ids):
            return [self.imgs[idx] for idx in ids]
        elif type(ids) is int:
            return [self.imgs[ids]]

    def show_cat_anns(self, cat_id=None, img_in=None, ann_in=None):
        # according cat_id or cat_id&img_ids show picture with gt bbox

        if not img_in:
            img_ids = self.get_img_ids(cat_id)
        else:
            img_ids = [0]
        for img_id in img_ids:
            if not img_in:
                img_path = os.path.join(self.data_dir,
                                        _remote_to_local(self.imgs[img_id]['image_url']))
            else:
                img_path = os.path.join(self.data_dir,
                                        _remote_to_local(img_in['image_url']))
            img = imread(img_path)
            plt.imshow(img)

            if not ann_in:
                ann_ids = self.get_ann_ids(cat_id, img_id)
            else:
                ann_ids = [0]
            ax = plt.gca()
            for ann_id in ann_ids:
                color = np.random.rand(3)
                if not ann_in:
                    ann = self.anns[ann_id]
                else:
                    ann = ann_in
                ax.add_patch(Rectangle((ann['x'], ann['y']),
                                       ann['w'],
                                       ann['h'],
                                       fill=False,
                                       edgecolor=color,
                                       linewidth=3))
                ax.text(ann['x'], ann['y'],
                        'name: ' + ann['names'][0],
                        style='italic',
                        size='larger',
                        bbox={'facecolor': 'white', 'alpha': .5})
                ax.text(ann['x'], ann['y'] + ann['h'],
                        'synsets: ' + ','.join(ann['synsets']),
                        style='italic',
                        size='larger',
                        bbox={'facecolor': 'white', 'alpha': .5})
            plt.show()

    def compute_cat_stats(self, full=False):
        ann_lens, img_lens = {}, {}
        for cnt, cat_id in enumerate(self.cats, 1):
            ann_lens[cat_id] = len(self.get_ann_ids(cat_id))
            img_lens[cat_id] = len(self.get_img_ids(cat_id))
            if cnt % 10 == 0:
                print('{} categories computed...'.format(cnt))
            elif cnt == len(self.cats):
                print('{} categories computed...'.format(cnt))

        self.ann_lens = sorted(ann_lens.items(),
                               key=lambda k_v: -k_v[1])
        self.img_lens = sorted(img_lens.items(),
                               key=lambda k_v: -k_v[1])
        if full:
            with open(os.path.join(self.data_dir, 'ann_lens_1000.txt'), 'w') as f:
                f.write('{},{},{}\n'.format('synset', 'category_id', '#instances'))
                for cat in self.ann_lens:
                    f.write('{},{},{}\n'.format(self.cats[cat[0]], cat[0], cat[1]))

            # with open(os.path.join(self.data_dir, 'img_lens.txt'), 'w') as f:
            #     f.write('{},{},{}\n'.format('synset', 'category_id', '#images'))
            #     for cat in self.img_lens:
            #         f.write('{},{},{}\n'.format(self.cats[cat[0]], cat[0], cat[1]))

        # cat_ids, ann_lens = zip(*self.ann_lens)
        # cats = [self.cats[cat_id].split('.')[0] for cat_id in cat_ids]
        # plt.subplot(2, 1, 1)
        # plt.bar(range(cnt), ann_lens, tick_label=cats)
        # plt.title('#Instances Per Category')
        #
        # cat_ids, img_lens = zip(*self.img_lens)
        # cats = [self.cats[cat_id].split('.')[0] for cat_id in cat_ids]
        # plt.subplot(2, 1, 2)
        # plt.bar(range(cnt), img_lens, tick_label=cats)
        # plt.title('#Images Per Category')
        # plt.show()

    def draw_synset_graph(self, ann_ids):
        """
        draw synsets in an image

        :param objects: objects (synsets) need to be drawn
        """
        synsets = []
        for ann_id in ann_ids:
            object = self.anns[ann_id]
            if len(object['synsets']) > 0:
                synsets += [wn.synset(synset) for synset in object['synsets']]
        graph = _construct_graph(synsets)
        colors = []
        for node in graph:
            if node in map(lambda x: x.name(), synsets):
                colors.append('r')
            elif node in ['entity.n.01']:
                colors.append('g')
            else:
                colors.append('b')
        nx.draw_networkx(graph, pos=gl(graph), node_color=colors)
        plt.tick_params(labelbottom='off', labelleft='off')
        # plt.show()
        plt.savefig("cls_synset.png")
        gc.collect()

    def get_major_ids(self, list, num=1000):
        sorted_cat_ids = np.loadtxt(
            os.path.join(self.data_dir, list),
            dtype=np.int32, delimiter=',', skiprows=1, usecols=1)
        return sorted_cat_ids[:num].tolist()

    def dump_train_val(self, val_num=5000):
        cat_ids = self.get_major_ids('ann_lens.txt')
        img_ids = self.get_img_ids(cat_ids)
        print('{} out of {} images are left for train/val'.
              format(len(img_ids), len(self.imgs)))
        for img_id in img_ids:
            self.imgs[img_id]['objects'] = \
                [object for object in self.imgs[img_id]['objects'] if
                 'category_id' in object and object['category_id'] in cat_ids]
        img_ids = np.array(img_ids)
        val_ids = set(np.random.choice(img_ids, val_num, False).tolist())
        assert len(val_ids) == val_num
        img_ids = set(img_ids.tolist())
        train_ids = img_ids - val_ids
        assert len(train_ids) + len(val_ids) == len(img_ids)
        train_imgs = [self.imgs[img_id] for img_id in train_ids]
        val_imgs = [self.imgs[img_id] for img_id in val_ids]

        with open(os.path.join(data_dir, 'objects_train.json'), 'w') as ft:
            json.dump(train_imgs, ft)
        with open(os.path.join(data_dir, 'objects_val.json'), 'w') as fv:
            json.dump(val_imgs, fv)

    def load_res(self, res_dir, res_file):
        return VG(res_dir, res_file)
def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train classifier network')
def norm_distribution(prob, num=1325,dim=1):
    num=prob.shape[1]
    prob=prob
    prob_weight = prob[:, :num]
    sum_value = np.sum(prob_weight, keepdims=True, axis=dim)
    prob_weight = prob_weight / np.repeat(sum_value, prob_weight.shape[dim], axis=dim)
    return prob_weight
def cp_kl(a, b):
    # compute kl diverse
    if np.sum(a) == 0 or np.sum(b) == 0:
        return 1
    sum_ = a * np.log(a / b)
    all_value = [x for x in sum_ if str(x) != 'nan' and str(x) != 'inf']
    kl = np.sum(all_value)
    return kl
def compute_js(attr_prob):
    cls_num = attr_prob.shape[0]
    similarity = np.zeros((cls_num, cls_num))

    for i in range(0, cls_num):
        if i % 50 == 0:
            print('had proccessed {} cls...\n'.format(i))
        for j in range(0, cls_num):
            if i == j:
                similarity[i,j] = 0
            else:
                similarity[i,j] = 0.5 * (cp_kl(attr_prob[i, :], 0.5*(attr_prob[i, :] + attr_prob[j,:]))
                                         + cp_kl(attr_prob[j, :], 0.5*(attr_prob[i, :] + attr_prob[j, :])))
    np.save('stastic_numpy', similarity)
    return  similarity
def multiprocess_js(group_array_,original_array,col_number,similarity,indexes):
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
                similarity[i, j] = 0.5 * (cp_kl(original_array[i, :], 0.5 * (original_array[i, :] + original_array[j, :]))
                                          + cp_kl(original_array[j, :], 0.5 * (original_array[i, :] + original_array[j, :])))
    return similarity[int(indexes[index]): int(indexes[index]+(indi_cls_num)),:]

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
def compute_custom_js(attr_prob,outfile):
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
    pool_result =pool.map(partial(multiprocess_js, original_array=attr_prob,col_number=cls_num,similarity=similarity,indexes=indexes),enumerate(group_attr_prob))
    pool.close()
    pool.join()
    start_line=0
    # fuck=np.zeros((10,10,1234))
    for line, result in enumerate(pool_result):
        similarity[int(indexes[line]):int(indexes[line]+result.shape[0]), :] = result
        start_line=start_line + group_attr_prob[line].shape[0]
    # for line, result in enumerate(fuck):
    #     similarity[line:line+fuck.shape[0], :] = result

    np.save(outfile, similarity)
    return  similarity
def compute_Wasserstein_distance(attr_prob,outfile):
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

    np.save(outfile, similarity)
    return  similarity
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from sklearn.datasets import load_iris
from sklearn.cluster import AgglomerativeClustering


def plot_dendrogram(model,distance, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix,leaf_font_size=9,distance_sort=True,leaf_rotation=90, show_leaf_counts=True,p=-1,**kwargs)

def filter_obj_rel(vg):#obj stastic is 7600 size
    obj_stastic=vg.rel_to_cat.values()
    obj_stastic=np.array(list(obj_stastic))
    rel_stastic = vg.cat_to_rel.values()
    rel_stastic = np.array(list(rel_stastic))
    obj_stastic_sum=obj_stastic.sum(axis=0)
    objsort_index=np.argsort((-1)*obj_stastic_sum)
    rel_stastic_sum = rel_stastic.sum(axis=0)
    relsort_index = np.argsort((-1)*rel_stastic_sum)
    return  obj_stastic_sum,objsort_index,rel_stastic_sum,relsort_index

def change_predicate(predicate):
    try:

        finallist = []

        filter_predicate=filter(predicate, useless_prefix,single_useless)
        if filter_predicate=='':
            return ''
        words = filter_predicate.split(' ')
        for pre in words:
            if pre=='ON':
                pre='on'
            if pre=='IN':
                pre='in'
            if pre == 'WEARING':
                pre = 'wearing'




            discard_word = []

            parsed_str = nlp.annotate((pre), properties=props)
            parsed_dict = json.loads(parsed_str)
            if len(words) != 1 and len(words) != 0:#


                for index, word in enumerate(parsed_dict['sentences'][0]['tokens']):
                    if ('VB' in word['pos']) and word['lemma']!='be':
                        pass
                    else:
                        discard_word.append(index)
                        # parsed_dict['sentences'][0]['tokens'].remove(word)
                [parsed_dict['sentences'][0]['tokens'].pop(i) for i in discard_word[::-1]]
                if len(parsed_dict['sentences'][0]['tokens']) == 0 or len(parsed_dict['sentences'][0]['tokens']) >= 2:
                    del parsed_dict['sentences'][0]['tokens']
                    continue

                lemma_list = [v for d in parsed_dict['sentences'][0]['tokens'] for k, v in d.items() if k == 'lemma']#['throw']
                finallist.append(lemma_list)
            if len(words) == 1 and len(words[0])!=1:#that is shouldn't occur single a b c

                for index, word in enumerate(parsed_dict['sentences'][0]['tokens']):
                    # print(parsed_dict['sentences'][0]['tokens'][0]['lemma'])
                    # if [parsed_dict['sentences'][0]['tokens'][0]['lemma']] == 'in':
                    #     print(',,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,')
                    if ('VB' in  word['pos'])  or ('IN' in word['pos']):
                        # if 'NN'!=word['pos']:
                        finallist.append([parsed_dict['sentences'][0]['tokens'][0]['lemma']])


                    else:
                        pass
                        # del parsed_dict['sentences'][0]['tokens']

        for index, prelist in enumerate(finallist):
            (finallist[index]) = (finallist[index][0])
        finallist = (list(finallist))
        if len(finallist)==0:
            return ''
        else:
            str = " ".join(finallist)
            return  str
    except:
        print('f')
        s1 = pickle.dumps(sys.exc_info())
        reraise(*pickle.loads(s1))
def multiprocess_rewrite_anno(dataset, index_array):
    jsonfile = list(dict())
    imgnew = dict()
    anns = []
    try:
        for cnt, img in enumerate(dataset,1):  # img:'imageid' 'objects' 'imageurl' 'hight' 'weight'
            imgnew['image_id'] = img['image_id']
            # this is relation anno
            for ann in img['relationships']:
                new_predicate = change_predicate(ann['predicate'])
                ann['new_pre'] = new_predicate
                ann['image_id'] = img['image_id']
                synsets_rel = ann['synsets']
                anns.append(ann)
            imgnew['relationships'] = anns
            anns = []
            if (len(imgnew['relationships']) != 0):
                jsonfile.append(imgnew)
            imgnew = {}
        return  jsonfile
    except:
        print('shit')
        s = pickle.dumps(sys.exc_info())
        reraise(*pickle.loads(s))



'''this function include filter useless word and nlp analyze, finally cahnge json'''
def rewrite_anno(annotation_file=None):

    split_dataset=[]
    if annotation_file is not None:
        print('loading ori annotations into memory...')
        dataset = json.load(open(os.path.join(
            annotation_file), 'r'))
    indexes = np.zeros((MAX_PROCESS, 1))
    temp = iter(dataset)
    for i in range(1,MAX_PROCESS):
        indexes[i]=len(dataset)//MAX_PROCESS + indexes[i-1]#important
        split_dataset.append(list(islice(temp, 0, int(indexes[i]-indexes[i-1]))))
    split_dataset.append(list(islice(temp, 0, int(len(dataset)-indexes[-1]))))

    # multiprocessing.set_start_method('spawn')
    pool = Pool(processes=MAX_PROCESS)

    res = [list(islice(temp, 0, MAX_PROCESS))  for ele in indexes]
    if (annotation_file.find('rel') != -1):
        pool_result = pool.map(
            partial(multiprocess_rewrite_anno, index_array= indexes ),
            (split_dataset))
        pool.close()
        pool.join()
        for result in pool_result:
            if isinstance(result, ExceptionWrapper):
                result.re_raise()
    jsonfile=[]

    for a in pool_result:
        jsonfile.extend(a)


    with open('relationship_refine.json', 'wb') as handle:
        handle.write(orjson.dumps(jsonfile))

def filter(predicates,useless_prefix,single_useless):

    # picklestr=[predicate.split() for   predicate in predicates] #use for when predicate is a list
    picklelist=[predicates.split()] #[['with']]
    for index, prelist in enumerate(picklelist):
        need_to_remove_index=[]
        # if len(prelist)>1:
        #     for ind, pre in enumerate(prelist):
        #         if pre in useless_prefix:
        #             need_to_remove_index.append(ind)
        #     [picklestr[index].pop(i) for i in need_to_remove_index[::-1]]
        #
        if len(prelist) ==1:
            if len(prelist[0])==1 or (prelist[0] in single_useless): #['ON']
                picklelist[index].remove(prelist[0])
        (picklelist[index]) = tuple(picklelist[index]) #before op:[['coming', 'from']]
        #     print(prelist)
    # picklelist=(tuple(picklelist))
    # picklelist=set(picklelist)

    # picklelist = (list(picklelist))
    for index, prelist in enumerate(picklelist):
        (picklelist[index]) = ' '.join(picklelist[index])
    for i,a in enumerate(picklelist): #there is 'on' behind
        a=a.lower()
        picklelist[i] = a
        # if 'ON' in a:
        #     if(len(a)==1):
        #         picklelist[i]='on'
        # if 'IN' in a:
        #     if (len(a) == 1):
        #         picklelist[i] = 'in'


    return  ' '.join(picklelist[:])#return str
'''this is only used for given a pickle with relationship,than analyze'''
def predicate_list(pickle):
    finallist = []
    for w in pickle:
        discard_word = []
        str = " ".join(w[:])
        parsed_str = nlp.annotate((str), properties=props)
        parsed_dict = json.loads(parsed_str)
        if len(w) != 1 and len(w) != 0:

            for index, word in enumerate(parsed_dict['sentences'][0]['tokens']):
                if 'VB' in word['pos']:
                    pass
                else:
                    discard_word.append(index)
                    # parsed_dict['sentences'][0]['tokens'].remove(word)
            [parsed_dict['sentences'][0]['tokens'].pop(i) for i in discard_word[::-1]]
            if len(parsed_dict['sentences'][0]['tokens']) == 0 or len(parsed_dict['sentences'][0]['tokens']) >= 2:
                del parsed_dict['sentences'][0]['tokens']
                continue

            lemma_list = [v for d in parsed_dict['sentences'][0]['tokens'] for k, v in d.items() if k == 'lemma']
            finallist.append(lemma_list)
        if len(w) == 1:

            for index, word in enumerate(parsed_dict['sentences'][0]['tokens']):
                # print(parsed_dict['sentences'][0]['tokens'][0]['lemma'])
                # if [parsed_dict['sentences'][0]['tokens'][0]['lemma']] == 'in':
                #     print(',,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,')
                if ('VB' in word['pos']) or ('IN' in word['pos']):
                    # if 'NN'!=word['pos']:
                    finallist.append([parsed_dict['sentences'][0]['tokens'][0]['lemma']])


                else:
                    del parsed_dict['sentences'][0]['tokens']

    for index, prelist in enumerate(finallist):
        (finallist[index]) = (finallist[index][0])
    finallist = (tuple(finallist))
    finallist = set(finallist)
    finallist = (list(finallist))
    for i, a in enumerate(finallist):
        finallist[i] = a[:]
    return  finallist

def map_predicate2cluster(predicate,cluster):
     index=range(3,8)
     predicate2cluster={}
     predicatename2cluster = {}
     for _ in range(0,5):
         predicate2cluster[str(predicate[_])] = str(index[_])
         predicatename2cluster[str(vg.cats[int(predicate[_])])] = str(index[_])
     for ids,pre in enumerate(predicate[5:]):

         predicate2cluster[str(pre)]=str(cluster[ids])
         predicatename2cluster[str(vg.cats[int(pre)])]=str(cluster[ids])
     return  predicate2cluster,predicatename2cluster

'''draw bar of object distribution'''
def plot_distribution_bar(rel_array,relsort_index): #input should be an np array
    stastic_photo_dir='stastic_photo_dir/'
    for predicates in range(rel_array.shape[0]) :

        objects = (rel_array[predicates,:])
        xs = np.arange(len(objects))
        fig = plt.figure(figsize=(10, 5))
        plt.bar(xs, objects, color='maroon',width=0.4)
        # plt.xticks(xs, labels)  # Replace default x-ticks with xs, then replace xs with labels
        # plt.yticks(ys)
        predicates = vg.cats[int(relsort_index[predicates])]
        plt.xlabel(predicates)
        plt.show()
        plt.savefig(stastic_photo_dir+str(predicates)+'.png')

def create_new_vg_sgg_h5py(vg_sgg, attributes, name='VG-SGG-with-attri.h5'):
    vg_sgg_with_attri = h5py.File(name, 'w')
    # copy from original vg_sgg
    for key in list(vg_sgg.keys()):
        vg_sgg_with_attri.create_dataset(key, data=vg_sgg[key][:])
    # add attributes
    vg_sgg_with_attri.create_dataset('attributes', data=attributes)
    vg_sgg_with_attri.close()
import h5py

if __name__ == '__main__':
    filename = 'datasets/vg/' + r'VG-SGG-dicts-with-attri.json'

    with h5py.File(filename, "r") as f:
        # List all groups
        print("Keys: %s" % f.keys())
        a_group_key = list(f.keys())[0]

        # Get the data
        data = list(f[a_group_key])
    # str='t wearing kid pet bank sideways fro idestort'
    # parsed_str = nlp.annotate((str), properties=props)
    # parsed_dict = json.loads(parsed_str)
    # parsed_str = nlp.annotate(('covered with'), properties=props)
    # parsed_str = nlp.annotate(('WEARING'), properties=props)
    single_useless=['and','of','from','for','at','with','to'] #this is for delete when there is single word in ori anno
    data_dir='datasets/vg/'#'/datasets/vg/'不能加最前面的/是因为会被误认为是绝对路径
    dataset_dir='/dataset/VG/image/'
    storage_dir='distance_result/'
    args = parse_args()
    imdb_name = "vg_rel"
    vg = VG(dataset_dir, annotation_file=data_dir + r'VG-SGG-dicts-with-attri.json')

    # rewrite_anno(annotation_file=data_dir + r'relationships.json')
    with open('predicate','rb') as pickle_file:
        predicate=pickle.load(pickle_file)
    picklestr=",".join(predicate.values())
    picklelist = picklestr.split(',')
    finallist=predicate_list(picklelist)


    obj_stastic_sum,objsort_index,rel_stastic_sum,relsort_index=filter_obj_rel(vg)
    with open('predicate_stastic', 'rb')  as readfile: #key is object category
        e=pickle.load(readfile)
    e_list=list(e.values())
    e_array = np.array(e_list)
    with open('object_stastic_no_syn', 'rb')  as readfile:# key is object predicate
        predicate_distribution=pickle.load(readfile)

    predicate_distribution_list=list(predicate_distribution.values())
    predicate_distribution_array = np.array(predicate_distribution_list)
    predicate_distribution_array=predicate_distribution_array[relsort_index[:50],:]
    predicate_distribution_array = predicate_distribution_array[:, objsort_index[:150]]
    relname=[]
    objname = []
    for x in relsort_index[:50]:
        relname.append(vg.cats[int(x)])
    for x in objsort_index[:150]:
        objname.append(list(vg.objects.keys())[x])
    with open('relation_name.json', 'w') as f:
        json.dump(relname, f)
    with open('object_name.json', 'w') as f:
        json.dump(objname, f)
    plot_distribution_bar(predicate_distribution_array,relsort_index)
    # if(os.path.exists(storage_dir+'stastic_numpy_custom.npy')==False):
    #     output_file=storage_dir+'stastic_numpy_custom.npy'
    #     graph_a=cout_w(e_array)
    #     graph_a = compute_custom_js(graph_a,output_file)
    # else:
    #     graph_a=np.load('stastic_numpy_custom.npy')
    #     a=1

    output_file=storage_dir+'m_distance.npy'
    graph_predicate_distribution = norm_distribution(predicate_distribution_array)
    graph_predicate_distribution = compute_Wasserstein_distance(graph_predicate_distribution,output_file)

    # agg = AgglomerativeClustering(  n_clusters=300,affinity='precomputed',
    #                               linkage='average',compute_full_tree=True,compute_distances=True)
    # u = agg.fit(graph_o)
    # print(u)
    from sklearn_extra.cluster import KMedoids
    kmedoids = KMedoids(n_clusters=3,metric='precomputed',method='pam',init='random',max_iter=100000).fit(graph_predicate_distribution[5:,5:])
    predicate2cluster=map_predicate2cluster(relsort_index[:50],kmedoids.labels_)
    with open(os.path.join(data_dir, 'predicate2cluster.json'), 'w') as ft:
        json.dump(predicate2cluster, ft)

    a=0
    # plt.figure(figsize=(300, 50))
    # plt.title('Hierarchical Clustering Dendrogram')
    # # plot the top three levels of the dendrogram
    # plot_dendrogram(u,graph_a,truncate_mode='level')
    # plt.xlabel("Number of points in node (or index of point if no parenthesis).")
    #
    # plt.tight_layout()
    # plt.gcf().autofmt_xdate()
    # # plt.show()
    #
    # plt.savefig('fuck.png')
    #
    #
    # a=1
    graph_1 = np.load('object_numpy_custom_kl.npy')
    graph_2 = np.load('object_numpy_kl.npy')
    graph_3 = np.load('object_numpy_js.npy')
    kmedoids1 = KMedoids(n_clusters=10, metric='precomputed', method='pam', init='random', max_iter=20000).fit(graph_1)
    kmedoids2 = KMedoids(n_clusters=60, metric='precomputed', method='pam', init='random', max_iter=20000).fit(graph_2)
    kmedoids3 = KMedoids(n_clusters=60, metric='precomputed', method='pam', init='random', max_iter=20000).fit(graph_3)

'''draw bar of object distribution'''
def plot_distribution_bar(rel_array): #input should be an np array
    stastic_photo_dir='stastic_photo_dir/'
    for predicates in range(rel_array.size(0)) :
        predicates=vg.cats[int(predicates)]
        objects = list(rel_array[predicates,:])
        xs = np.arange(len(objects))
        fig = plt.figure(figsize=(10, 5))
        plt.bar(predicates, objects, color='maroon',width=0.4)
        # plt.xticks(xs, labels)  # Replace default x-ticks with xs, then replace xs with labels
        # plt.yticks(ys)
        plt.show()
        plt.savefig(stastic_photo_dir,str(predicates)+'.png')

