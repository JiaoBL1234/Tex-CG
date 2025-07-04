from PIL import Image, ImageFile

from torch.utils.data import Dataset
import os.path as osp
import random
import torch
ImageFile.LOAD_TRUNCATED_IMAGES = True


def read_image(img_path):
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img


class BaseDataset(object):

    def get_imagedata_info(self, data):
        pids, cams, tracks, clothids = [], [], [], []
        for _, pid, camid, trackid, clothid in data:
            pids += [pid]
            cams += [camid]
            tracks += [trackid]
            clothids += [clothid]
        pids = set(pids)
        cams = set(cams)
        tracks = set(tracks)
        clothids = set(clothids)
        num_pids = len(pids)
        num_cams = len(cams)
        num_imgs = len(data)
        num_views = len(tracks)
        num_cloth_ids = len(clothids)
        return num_pids, num_imgs, num_cams, num_views, num_cloth_ids

    def print_dataset_statistics(self):
        raise NotImplementedError


class BaseImageDataset(BaseDataset):

    def print_dataset_statistics(self, train, query, gallery, query_cc=None, gallery_cc=None):
        num_train_pids, num_train_imgs, num_train_cams, num_train_views, num_train_clothids = self.get_imagedata_info(train)
        num_query_pids, num_query_imgs, num_query_cams, num_train_views, num_query_clothids = self.get_imagedata_info(query)
        num_gallery_pids, num_gallery_imgs, num_gallery_cams, num_train_views, num_gallery_clothids = self.get_imagedata_info(gallery)
        if query_cc:
            num_query_cc_pids, num_query_cc_imgs, num_query_cc_cams, num_train_views, num_query_cc_clothids = self.get_imagedata_info(query_cc)
        if gallery_cc:
            num_gallery_cc_pids, num_gallery_cc_imgs, num_gallery_cc_cams, num_train_views, num_gallery_cc_clothids = self.get_imagedata_info(gallery_cc)


        print("Dataset statistics:")
        print("  ---------------------------------------------------")
        print("  subset   | # ids | # images | # cameras | # clothes ")
        print("  ---------------------------------------------------")
        print("  train    | {:5d} | {:8d} | {:9d}| {:9d}".format(num_train_pids, num_train_imgs, num_train_cams, num_train_clothids))
        print("  query    | {:5d} | {:8d} | {:9d}| {:9d}".format(num_query_pids, num_query_imgs, num_query_cams, num_query_clothids))
        print("  gallery  | {:5d} | {:8d} | {:9d}| {:9d}".format(num_gallery_pids, num_gallery_imgs, num_gallery_cams, num_gallery_clothids))
        if query_cc:
            print("  query_cc | {:5d} | {:8d} | {:9d}| {:9d}".format(num_query_cc_pids, num_query_cc_imgs, num_query_cc_cams, num_query_cc_clothids))
        if gallery_cc:
            print(" gallery_cc| {:5d} | {:8d} | {:9d}| {:9d}".format(num_gallery_cc_pids, num_gallery_cc_imgs, num_gallery_cc_cams, num_gallery_cc_clothids))
        print("  ---------------------------------------------------")


class ImageDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid, trackid, cloth_id = self.dataset[index]
        img = read_image(img_path)

        if self.transform is not None:
            img = self.transform(img)

        return img, pid, cloth_id, camid, trackid, img_path.split('/')[-1]