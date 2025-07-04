import glob
import re

import os.path as osp

from .bases import BaseImageDataset
from collections import defaultdict
import pickle
import numpy as np 
import random
import pdb

class PRCC(BaseImageDataset):
    dataset_dir = 'PRCC'

    def __init__(self, root='', verbose=True, pid_begin = 0, **kwargs):
        super(PRCC, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'rgb/train')
        self.test_dir = osp.join(self.dataset_dir, 'rgb/test')
        self._check_before_run()

        train, num_train_pids, num_train_imgs, num_train_clothids, pid2clothes = self._process_dir_train(self.train_dir)



        query_same, query_cc, gallery, num_test_pids, num_test_clothids, \
        num_query_imgs_same, num_query_imgs_diff, num_gallery_imgs = self._process_dir_test(self.test_dir)


        self.pid_begin = 0

        if verbose:
            print("=> PRCC loaded")
            self.print_dataset_statistics(train, query_same, gallery, query_cc, gallery)

        self.train_pid2clothes = pid2clothes
        self.train = train
        self.query = query_same
        self.gallery = gallery
        self.query_cc = query_cc
        self.gallery_cc = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids, self.num_train_clothids = num_train_pids, num_train_imgs, 3, 0, num_train_clothids
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids, self.num_test_clothids = num_test_pids, num_query_imgs_same, 3, 0, num_test_clothids
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids, self.num_gallery_clothids = num_test_pids, num_gallery_imgs, 3, 0, num_test_clothids
        self.num_query_cc_pids, self.num_query_cc_imgs, self.num_query_cc_cams, self.num_query_cc_vids, self.num_query_cc_clothids = num_test_pids, num_query_imgs_diff, 3, 0, num_test_clothids
        

    def _check_before_run(self):
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.test_dir):
            raise RuntimeError("'{}' is not available".format(self.test_dir))


    def _process_dir_train(self, dir_path):
        pdirs = glob.glob(osp.join(dir_path, '*'))
        pdirs.sort()

        pid_container = set()
        clothes_container = set()
        for pdir in pdirs:
            pid = int(osp.basename(pdir))
            pid_container.add(pid)
            img_dirs = glob.glob(osp.join(pdir, '*.jpg'))
            for img_dir in img_dirs:
                cam = osp.basename(img_dir)[0]  
                if cam in ['A', 'B']:
                    clothes_container.add(osp.basename(pdir))
                else:
                    clothes_container.add(osp.basename(pdir) + osp.basename(img_dir)[0])
        pid_container = sorted(list(pid_container))
        clothes_container = sorted(list(clothes_container))
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        clothes2label = {clothes_id: label for label, clothes_id in enumerate(clothes_container)}
        cam2label = {'A': 0, 'B': 1, 'C': 2}

        num_pids = len(pid_container)
        num_clothes = len(clothes_container)

        dataset = []
        pid2clothes = np.zeros((num_pids, num_clothes))
        for pdir in pdirs:
            pid = int(osp.basename(pdir))
            img_dirs = glob.glob(osp.join(pdir, '*.jpg'))

            for img_dir in img_dirs:
                cam = osp.basename(img_dir)[0]  
                label = pid2label[pid]
                camid = cam2label[cam]
                if cam in ['A', 'B']:
                    clothes_id = clothes2label[osp.basename(pdir)]
                else:
                    clothes_id = clothes2label[osp.basename(pdir) + osp.basename(img_dir)[0]]

                dataset.append((img_dir, label, camid, 0, clothes_id))
                pid2clothes[label, clothes_id] = 1

        num_imgs = len(dataset)

        return dataset, num_pids, num_imgs, num_clothes, pid2clothes
    
    
    def _process_dir_test(self, test_path):

        pid_container = set()
        for pdir in glob.glob(osp.join(test_path, 'A', '*')):
            pid = int(osp.basename(pdir))
            pid_container.add(pid)
        pid_container = sorted(pid_container)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        cam2label = {'A': 0, 'B': 1, 'C': 2}

        num_pids = len(pid_container)
        num_clothes = num_pids * 2

        query_dataset_same_clothes = []
        query_dataset_diff_clothes = []
        gallery_dataset = []

        pdirs = glob.glob(osp.join(test_path, 'A', '*'))
        pdirs.sort()
        camid = cam2label['A']
        for pdir in pdirs:
            pid = pid2label[int(osp.basename(pdir))]
            img_dirs = glob.glob(osp.join(pdir, '*.jpg'))
            for img_dir in sorted(img_dirs):
                clothes_id = pid * 2
                gallery_dataset.append((img_dir, pid, camid, 0, clothes_id))
            

        for cam in ['B', 'C']:
            camid = cam2label[cam]
            pdirs = glob.glob(osp.join(test_path, cam, '*'))
            pdirs.sort()
            for pdir in pdirs:
                pid = pid2label[int(osp.basename(pdir))]
                img_dirs = glob.glob(osp.join(pdir, '*.jpg'))
                for img_dir in sorted(img_dirs):
                    if cam == 'B':
                        clothes_id = pid * 2
                        query_dataset_same_clothes.append((img_dir, pid, camid, 0, clothes_id))
                    else:
                        clothes_id = pid * 2 + 1
                        query_dataset_diff_clothes.append((img_dir, pid, camid, 0, clothes_id))



        num_imgs_query_same = len(query_dataset_same_clothes)
        num_imgs_query_diff = len(query_dataset_diff_clothes)
        num_imgs_gallery = len(gallery_dataset)

        return query_dataset_same_clothes, query_dataset_diff_clothes, gallery_dataset, \
               num_pids, num_clothes, num_imgs_query_same, num_imgs_query_diff, num_imgs_gallery
