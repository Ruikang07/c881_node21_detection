import random
import torch
import collections
import numbers
import numpy as np

from torchvision.transforms import functional as F
from torchvision.transforms import InterpolationMode


def _flip_coco_person_keypoints(kps, width):
    flip_inds = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
    flipped_data = kps[:, flip_inds]
    flipped_data[..., 0] = width - flipped_data[..., 0]
    # Maintain COCO convention that if visibility == 0, then x, y = 0
    inds = flipped_data[..., 2] == 0
    flipped_data[inds] = 0
    return flipped_data


def _flip_coco_person_keypoints_v(kps, height):
    flip_inds = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
    flipped_data = kps[:, flip_inds]
    flipped_data[..., 1] = height - flipped_data[..., 1]
    # Maintain COCO convention that if visibility == 0, then x, y = 0
    inds = flipped_data[..., 2] == 0
    flipped_data[inds] = 0
    return flipped_data


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(-1)
            bbox = target["boxes"]
            bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
            target["boxes"] = bbox
            if "masks" in target:
                target["masks"] = target["masks"].flip(-1)
            if "keypoints" in target:
                keypoints = target["keypoints"]
                keypoints = _flip_coco_person_keypoints(keypoints, width)
                target["keypoints"] = keypoints
        return image, target


class RandomVerticalFlip(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(0)
            bbox = target["boxes"]
            bbox[:, [1, 3]] = height - bbox[:, [3, 1]]
            target["boxes"] = bbox
            if "masks" in target:
                target["masks"] = target["masks"].flip(0)
            if "keypoints" in target:
                keypoints = target["keypoints"]
                keypoints = _flip_coco_person_keypoints_v(keypoints, height)
                target["keypoints"] = keypoints
        return image, target


class Resize(object):

    def __init__(self, size=1500, interpolation=InterpolationMode.BILINEAR):
        assert isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, image, target):

        image = F.resize(image, self.size, self.interpolation)
        bbox = (target["boxes"]).numpy()
        ratio = self.size/1024
        bbox = torch.from_numpy(np.trunc(bbox*ratio))
        target["boxes"] = bbox

        return image, target


class RandomCrop(object):
    """Crop the given PIL Image at a random location.

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
        padding (int or sequence, optional): Optional padding on each border
            of the image. Default is 0, i.e no padding. If a sequence of length
            4 is provided, it is used to pad left, top, right, bottom borders
            respectively.
    """

    def __init__(self, size=1024, padding=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding

    @staticmethod
    def get_params(image, output_size):

        w, h = image.size
        th, tw = output_size

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, image, target):

        if self.padding > 0:
            image = F.pad(image, self.padding)

        i, j, h, w = self.get_params(image, self.size)
        image = F.crop(image, i, j, h, w)

        th, tw = self.size
        bbox = (target["boxes"]).numpy()
        labels = (target['labels']).numpy()
        image_id = target["image_id"].numpy()
        area = target["area"].numpy()
        iscrowd = target["iscrowd"].numpy()
        delete_list = []

        for k in range(bbox.shape[0]):
            if bbox[k, 0] >= j + tw or bbox[k, 1] >= i + th or bbox[k, 2] <= j or bbox[k, 3] <= i:
                delete_list.append(k)
            else:
                if bbox[k, 0] < j:
                    bbox[k, 0] = 0
                else:
                    bbox[k, 0] = bbox[k, 0] - j

                if bbox[k, 1] < i:
                    bbox[k, 1] = 0
                else:
                    bbox[k, 1] = bbox[k, 1] - i

                if bbox[k, 2] > j + tw:
                    bbox[k, 2] = tw
                else:
                    bbox[k, 2] = bbox[k, 2] - j

                if bbox[k, 3] > i + th:
                    bbox[k, 3] = th
                else:
                    bbox[k, 3] = bbox[k, 3] - i

        #for m in range(len(delete_list)):
            #print('bbox.shape[0]: ', bbox.shape[0])
            #print('delete_list[m]: ', delete_list[m])
        bbox = np.delete(bbox, delete_list, axis=0)
        labels = np.delete(labels, delete_list)
        #image_id = np.delete(image_id, delete_list)
        area = np.delete(area, delete_list)
        iscrowd = np.delete(iscrowd, delete_list)

        target["boxes"] = torch.from_numpy(np.trunc(bbox))
        target["labels"] = torch.from_numpy(labels)
        target["image_id"] = torch.from_numpy(image_id)
        target["area"] = torch.from_numpy(area)
        target["iscrowd"] = torch.from_numpy(iscrowd)


        return image, target


class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target
