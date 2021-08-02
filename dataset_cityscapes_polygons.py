import json
import os
from collections import namedtuple

import torch
import torch.utils.data as data
from PIL import Image
import numpy as np
import logging
import sys

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')

def recursive_glob(rootdir=".", suffix=""):
    """Performs recursive glob with given suffix and rootdir
        :param rootdir is the root directory
        :param suffix is the suffix to be searched
    """
    return [
        os.path.join(looproot, filename)
        for looproot, _, filenames in os.walk(rootdir)
        for filename in filenames
        if filename.endswith(suffix)
    ]

class Cityscapes(data.Dataset):
    """Cityscapes <http://www.cityscapes-dataset.com/> Dataset.

    **Parameters:**
        - **root** (string): Root directory of dataset where directory 'leftImg8bit' and 'gtFine' or 'gtCoarse' are located.
        - **split** (string, optional): The image split to use, 'train', 'test' or 'val' if mode="gtFine" otherwise 'train', 'train_extra' or 'val'
        - **mode** (string, optional): The quality mode to use, 'gtFine' or 'gtCoarse' or 'color'. Can also be a list to output a tuple with all specified target types.
        - **transform** (callable, optional): A function/transform that takes in a PIL image and returns a transformed version. E.g, ``transforms.RandomCrop``
        - **target_transform** (callable, optional): A function/transform that takes in the target and transforms it.
    """

    # Based on https://github.com/mcordts/cityscapesScripts
    CityscapesClass = namedtuple('CityscapesClass', ['name', 'id', 'train_id', 'category', 'category_id',
                                                     'has_instances', 'ignore_in_eval', 'color'])
    classes = [
        CityscapesClass('unlabeled', 0, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('ego vehicle', 1, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('rectification border', 2, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('out of roi', 3, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('static', 4, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('dynamic', 5, 255, 'void', 0, False, True, (111, 74, 0)),
        CityscapesClass('ground', 6, 255, 'void', 0, False, True, (81, 0, 81)),
        CityscapesClass('road', 7, 0, 'flat', 1, False, False, (128, 64, 128)),
        CityscapesClass('sidewalk', 8, 1, 'flat', 1, False, False, (244, 35, 232)),
        CityscapesClass('parking', 9, 255, 'flat', 1, False, True, (250, 170, 160)),
        CityscapesClass('rail track', 10, 255, 'flat', 1, False, True, (230, 150, 140)),
        CityscapesClass('building', 11, 2, 'construction', 2, False, False, (70, 70, 70)),
        CityscapesClass('wall', 12, 3, 'construction', 2, False, False, (102, 102, 156)),
        CityscapesClass('fence', 13, 4, 'construction', 2, False, False, (190, 153, 153)),
        CityscapesClass('guard rail', 14, 255, 'construction', 2, False, True, (180, 165, 180)),
        CityscapesClass('bridge', 15, 255, 'construction', 2, False, True, (150, 100, 100)),
        CityscapesClass('tunnel', 16, 255, 'construction', 2, False, True, (150, 120, 90)),
        CityscapesClass('pole', 17, 5, 'object', 3, False, False, (153, 153, 153)),
        CityscapesClass('polegroup', 18, 255, 'object', 3, False, True, (153, 153, 153)),
        CityscapesClass('traffic light', 19, 6, 'object', 3, False, False, (250, 170, 30)),
        CityscapesClass('traffic sign', 20, 7, 'object', 3, False, False, (220, 220, 0)),
        CityscapesClass('vegetation', 21, 8, 'nature', 4, False, False, (107, 142, 35)),
        CityscapesClass('terrain', 22, 9, 'nature', 4, False, False, (152, 251, 152)),
        CityscapesClass('sky', 23, 10, 'sky', 5, False, False, (70, 130, 180)),
        CityscapesClass('person', 24, 11, 'human', 6, True, False, (220, 20, 60)),
        CityscapesClass('rider', 25, 12, 'human', 6, True, False, (255, 0, 0)),
        CityscapesClass('car', 26, 13, 'vehicle', 7, True, False, (0, 0, 142)),
        CityscapesClass('truck', 27, 14, 'vehicle', 7, True, False, (0, 0, 70)),
        CityscapesClass('bus', 28, 15, 'vehicle', 7, True, False, (0, 60, 100)),
        CityscapesClass('caravan', 29, 255, 'vehicle', 7, True, True, (0, 0, 90)),
        CityscapesClass('trailer', 30, 255, 'vehicle', 7, True, True, (0, 0, 110)),
        CityscapesClass('train', 31, 16, 'vehicle', 7, True, False, (0, 80, 100)),
        CityscapesClass('motorcycle', 32, 17, 'vehicle', 7, True, False, (0, 0, 230)),
        CityscapesClass('bicycle', 33, 18, 'vehicle', 7, True, False, (119, 11, 32)),
        CityscapesClass('license plate', -1, 255, 'vehicle', 7, False, True, (0, 0, 142)),
    ]

    train_id_to_color = [c.color for c in classes if (c.train_id != -1 and c.train_id != 255)]
    train_id_to_color.append([0, 0, 0])
    train_id_to_color = np.array(train_id_to_color)
    id_to_train_id = np.array([c.train_id for c in classes])
    # valid_train_id_to_name = [c.name for c in classes if (c.train_id!=255 and c.train_id != -1)]
    valid_train_id_to_name = ['person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle']

    # train_id_to_color = [(0, 0, 0), (128, 64, 128), (70, 70, 70), (153, 153, 153), (107, 142, 35),
    #                      (70, 130, 180), (220, 20, 60), (0, 0, 142)]
    # train_id_to_color = np.array(train_id_to_color)
    # id_to_train_id = np.array([c.category_id for c in classes], dtype='uint8') - 1

    def __init__(self, root, split='train', transform=None):
        self.root = os.path.expanduser(root)
        self.mode = 'gtFine'
        self.images_dir = os.path.join(self.root, 'leftImg8bit','train_ssl',split)

        self.targets_dir = os.path.join(self.root, 'gtFine', split)
        self.transform = transform

        self.split = split
        self.targets = recursive_glob(rootdir=self.targets_dir,suffix=".json")
        self.images = []

        if split not in ['train', 'test', 'val']:
            raise ValueError('Invalid split for mode! Please use split="train", split="test"'
                             ' or split="val"')

        if not os.path.isdir(self.images_dir) :
            raise RuntimeError('Dataset not found or incomplete. Please make sure all required folders for the'
                               ' specified "split" and "mode" are inside the "root" directory')

    @classmethod
    def valid_class_name(cls,):
        return cls.valid_train_id_to_name

    @classmethod
    def encode_target(cls, target):
        return cls.id_to_train_id[np.array(target)]

    @classmethod
    def decode_target(cls, target):
        target[target == 255] = 19
        # target = target.astype('uint8') + 1
        return cls.train_id_to_color[target]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is a tuple of all target types if target_type is a list with more
            than one item. Otherwise target is a json object if target_type="polygon", else the image segmentation.
        """
        baseName = os.path.basename(self.targets[index]).split('gt')[0]+'leftImg8bit.png'
        cityName = self.targets[index].split('/')[-2]
        image = Image.open(os.path.join(self.images_dir,cityName,baseName)).convert('RGB')
        logging.info(self.targets[index])
        dict_data = self.extractor(self.targets[index],image)
        # print(self.valid_class_name())
        # print(nmbs)
        # exit()
        # if self.transform:
        #     image = self.transform(image)
        return dict_data

    def __len__(self):
        return len(self.targets)

    def _load_json(self, path):
        with open(path, 'r') as file:
            data = json.load(file)
        return data

    def _get_target_suffix(self, mode, target_type):
        if target_type == 'instance':
            return '{}_instanceIds.png'.format(mode)
        elif target_type == 'semantic':
            return '{}_labelIds.png'.format(mode)
        elif target_type == 'color':
            return '{}_color.png'.format(mode)
        elif target_type == 'polygon':
            return '{}_polygons.json'.format(mode)
        elif target_type == 'depth':
            return '{}_disparity.png'.format(mode)

    def position(self, pos):
        # Find the xmin, ymin, xmax, ymax of selected object
        x = []
        y = []
        nums = len(pos)
        for i in range(nums):
            x.append(pos[i][0])
            y.append(pos[i][1])
        x_max = max(x)
        x_min = min(x)
        y_max = max(y)
        y_min = min(y)
        b = (float(x_min), float(y_min), float(x_max), float(y_max))
        return b

    def convert(self, size, box):
        # convert xmin, ymin, xmax, ymax to x,y,w,h
        dw = 1. / (size[0])
        dh = 1. / (size[1])
        x = (box[0] + box[1]) / 2.0 - 1
        y = (box[2] + box[3]) / 2.0 - 1
        w = box[1] - box[0]
        h = box[3] - box[2]
        x = x * dw
        w = w * dw
        y = y * dh
        h = h * dh
        return (x, y, w, h)

    def extractor(self, target, image):
        with open(target, 'r') as file:
            load_annotation = json.load(file)

        objects = load_annotation['objects']
        NameList = os.path.basename(target).split('_')

        nums = len(objects)
        valid_labels = []
        valid_polygon = []
        cropped_img_dict = {name:0 for name in self.valid_class_name()}

        for i in range(0, nums):
            labels = objects[i]['label']
            if labels in self.valid_class_name():
                valid_labels.append(labels)
                valid_polygon.append(objects[i]['polygon'])

        # logging.info(len(valid_labels))
        for i in range(len(valid_labels)):
            pos = valid_polygon[i]
            bbox = self.position(pos)
            bbox_w = bbox[2]-bbox[0]
            bbox_h = bbox[3]-bbox[1]
            # print('bbox_h,bbox_w->',bbox_h,bbox_w)
            if min(bbox_w,bbox_h)<50:
                continue
            else:
                merge_x = int(bbox_w * 15 /100)
                merge_y = int(bbox_h * 15 /100)
                image.crop((bbox[0]-merge_x,bbox[1]-merge_y,bbox[2]+merge_x,bbox[3]+merge_y)).save('./cropped_data/{}_{}_{}_{}_{}.png'.format(NameList[0],NameList[1],NameList[2],valid_labels[i],i))
                cropped_img_dict[valid_labels[i]] +=1

        logging.info(cropped_img_dict)
        return cropped_img_dict


if __name__ == '__main__':
    # import torchvision.transforms as transforms
    from collections import Counter

    data = Cityscapes(root='../MOCO_v2_r_2/data/cityscapes/')
    data_loader =  torch.utils.data.DataLoader(dataset=data,
                                               batch_size=1,
                                               shuffle=False, pin_memory=True, num_workers=0)
    logging.info('The numbers of train data is {}'.format(data_loader.__len__()))
    counter_classes = {}
    # counter_classes = Counter(counter_classes)
    for i, input in enumerate(data_loader):
        counter_classes =dict(Counter(counter_classes)+Counter(input))
        logging.info(counter_classes)
