import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
import random
import cv2
import traceback
from PIL import Image
import scipy.io as sio
import scipy.ndimage
from glob import glob
import json
import logging
import sys
import os

from utils.oct_dataset import RealDataset
from utils.point_transform import transform_3D_to_2D

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root)
from utils.hand_detector import calculate_com_2d, crop_area_3d
from utils.point_transform import transform_2D_to_3D

logging.basicConfig(level=logging.INFO, format="%(asctime)s: %(levelname)s %(name)s:%(lineno)d] %(message)s")
logger = logging.getLogger(__file__)


class OCTFeeder(Dataset):
    def __init__(self, phase='train'):
        """

        :param phase: train or test
        :param max_jitter:
        :param depth_sigma:
        """
        self.phase = phase
        config_file = os.path.join(root, "config", "dataset", "oct.json")
        self.config = json.load(open(config_file, 'r'))
        self.fx = self.config['camera']['fx']
        self.fy = self.config['camera']['fy']
        self.u0 = self.config['camera']['u0']
        self.v0 = self.config['camera']['v0']
        self.inter_matrix = np.array([
            [self.fx, 0, self.u0],
            [0, self.fy, self.v0],
            [0, 0, 1]
        ], dtype=np.float32)
        self.dataset = RealDataset(data_root=self.config['path'], labels_dir='all')
        self.cube = np.array(self.config["cube"], dtype=np.float32)
        self.joint_2d, self.joint_3d, self.depth_path = self.load_annotation()
        self.index = np.arange(len(self.depth_path))
        logger.info("{} num: {}".format(phase, len(self.index)))

    def load_annotation(self):
        indices = []
        if self.phase == 'train':
            for i in range(len(self.dataset)):
                if i % 5 != 0:
                    indices.append(i)
        else:
            for i in range(len(self.dataset)):
                if i % 5 == 0:
                    indices.append(i)

        joint_2d = []
        joint_3d = []
        depth_map_list = []
        for idx in indices:
            _, keypoints = self.dataset[idx]

            depth_path = self.dataset.get_volume_path(idx)
            depth_path = depth_path.replace(self.config['path'], os.path.join(self.config['path'], 'Depth'))
            depth_path = depth_path.replace(".npy", ".png")

            depth_map_list.append(depth_path)
            joint_3d.append(keypoints)
            joint_2d.append(transform_3D_to_2D(keypoints, self.fx, self.fy, self.u0, self.v0))

        return np.array(joint_2d, dtype=np.float32), np.array(joint_3d, dtype=np.float32), depth_map_list

    def __getitem__(self, item):
        item = self.index[item]
        joint_2d, joint_3d, depth_path = self.joint_2d[item], self.joint_3d[item], self.depth_path[item]

        try:
            depth = np.asarray(Image.open(depth_path), np.float32)
        except FileNotFoundError:
            return item, None, None, joint_3d, None, None, self.inter_matrix

        com_3d = np.mean(joint_3d, axis=0)

        com_2d = self.inter_matrix @ com_3d[:, None]
        com_2d = np.squeeze(com_2d)
        com_2d[:2] /= com_2d[2]
        com_2d = com_2d.astype(np.float32)

        cropped = cv2.resize(depth, (176, 176))
        crop_trans = np.eye(3, dtype=np.float32)  # TODO: this might not be correct

        return item, depth[None, ...], cropped[None, ...], joint_3d, \
            np.array(crop_trans), com_2d, self.inter_matrix, self.cube

    def __len__(self):
        return len(self.index)


if __name__ == '__main__':
    from tqdm import tqdm
    from feeders.nyu_feeder import collate_fn
    train_dataset = OCTFeeder('train')
    dataloader = DataLoader(train_dataset, shuffle=False, batch_size=4, collate_fn=collate_fn, num_workers=4)
    for batch_idx, batch_data in enumerate(tqdm(dataloader)):
        item, depth, cropped, joint_3d, crop_trans, com_2d, inter_matrix, cube = batch_data

    # print(item)
    # print(depth.shape)
    # print(cropped.shape)
    # print(joint_3d.shape)
    # print(crop_trans.shape)
    # print(com_2d.shape)
    # print(inter_matrix.shape)
    # print(cube.shape)
