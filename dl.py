"""
Support deeplake dataset
"""

# import os
# import xml.etree.ElementTree as ET
# import cv2

import deeplake
import numpy as np
from torch.utils.data import Dataset
import torch
from config import IMAGE_MEAN
from ctpn_utils import cal_rpn


class DeepLakeDataset(Dataset):
    def __init__(self, path):
        self.dataset = deeplake.load(path)

    def __len__(self):
        return self.dataset.__len__()

    def __getitem__(self, idx):
        img = self.dataset["image"][idx].numpy()
        # gtboxes: [[xmin, ymin, xmax, ymax], ...]
        gtboxes = self.dataset["boxes/box"][idx].data()

        # convert to numpy array
        # gtbox: [(xmin, ymin, xmax, ymax), ...]
        gtbox_list = []
        for box in gtboxes["value"]:
            gtbox_list.append((box[0], box[1], box[2], box[3]))
        gtbox = np.array(gtbox_list)

        # fuzhi zhantie de code
        h, w, c = img.shape
        # clip image
        if np.random.randint(2) == 1:
            img = img[:, ::-1, :]
            newx1 = w - gtbox[:, 2] - 1
            newx2 = w - gtbox[:, 0] - 1
            gtbox[:, 0] = newx1
            gtbox[:, 2] = newx2

        [cls, regr], _ = cal_rpn((h, w), (int(h / 16), int(w / 16)), 16, gtbox)

        m_img = img - IMAGE_MEAN

        regr = np.hstack([cls.reshape(cls.shape[0], 1), regr])

        cls = np.expand_dims(cls, axis=0)

        # transform to torch tensor
        m_img = torch.from_numpy(m_img.transpose([2, 0, 1])).float()
        cls = torch.from_numpy(cls).float()
        regr = torch.from_numpy(regr).float()

        return m_img, cls, regr


"""
Test codes
"""
if __name__ == "__main__":
    path = "hub://activeloop/icdar-2013-text-localize-train"

    dataset = DeepLakeDataset(path)
    print(dataset.__getitem__(0))
