import json
import os
import sys

import numpy as np
from torch.utils.data import Dataset


def rescale_keypoints(keypoints, keypoints_min, keypoints_max, target_shape):
    keypoints = (keypoints - keypoints_min) / (keypoints_max - keypoints_min)  # normalize keypoints to range [0, 1]
    return keypoints * target_shape  # scale keypoints to the target shape


class RealDataset(Dataset):
    def __init__(self, data_root, labels_dir=None, transform=None):
        self.data_root = data_root
        self.labels_dir = labels_dir
        self.transform = transform

        self.data = []
        self._load_data()

    def __getitem__(self, idx):
        volume_path = self.data[idx]
        return self.get_item(volume_path)

    def get_item(self, volume_path):
        if self.labels_dir is not None:
            volume_dir = os.path.dirname(volume_path)
            volume_name = os.path.basename(volume_path)
            metadata_path = os.path.join(volume_dir, self.labels_dir, volume_name.replace(".npy", ".json"))
        else:
            metadata_path = volume_path.replace(".npy", ".json")

        volume = np.load(volume_path) / 255.0
        volume = np.transpose(volume, (2, 0, 1))  # height, depth, width -> width, height, depth
        # volume = np.transpose(volume[0], (0, 2, 1))  # for full-size volumes
        volume = np.flip(volume, axis=1)

        with open(metadata_path, "r") as f:
            data_json = json.load(f)
            keypoints = np.array(data_json["keypoints"])
            keypoints = keypoints / data_json["volume_size"] * volume.shape

        if self.transform:
            return self.transform(volume, keypoints, os.path.basename(volume_path))
        return volume, keypoints

    def get_volume_path(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)

    def get_input_size(self):
        volume_shape = self.__getitem__(0)[0].shape
        return volume_shape[-3:]

    def _load_data(self):
        for root, dirs, files in os.walk(self.data_root):
            for file in files:
                if file.endswith('.npy'):
                    self.data.append(os.path.join(root, file))


if __name__ == '__main__':
    # Convert OCT volumes to depth maps #####################################

    from ops.render import point_cloud_mask_to_depth
    from PIL import Image

    def volume_to_depth(volume):
        volume[:, -10:, :] = 0  # Remove the noise at the top of volume

        width, height, depth = volume.shape
        depth_map = np.full((width, depth), height, dtype=np.float32)

        for x in range(width):
            for z in range(depth):
                for y in range(height - 1, -1, -1):
                    if volume[x, y, z] > 0:
                        depth_map[x, z] = height - y
                        break

        return depth_map


    def adjust_black_white_levels(data: np.ndarray, black_threshold: int,
                                  white_threshold: int) -> np.ndarray:
        """
        Apply Black and White level thresholds.
        For testing to see optimal thresholds,
        only supply a few slices to speed up process.
        Args:
            data (array): Input 3D numpy array data of type uint8.
            black_threshold (int): Black level threshold in decibels.
            white_threshold (int): White level threshold in decibels.

        Returns:
            array: Adjusted image array of type uint8.
        """

        image = data.astype(np.float32)

        # Apply black and white level threshold
        # Equivalent to but faster than np.minimum(white_threshold,
        #                               np.maximum(image, black_threshold))
        image = np.clip(image, black_threshold, white_threshold)

        # Normalize the thresholded image to [0, 1] range
        image = (image - black_threshold) / (white_threshold - black_threshold)

        # Convert the normalized image back to the original [0, 255] range

        image = image * 255
        image = image.astype(np.uint8)
        return image

    data_path = "C:/Data/OCT/ISMR OCT Data - Downsampled/"
    depth_path = "C:/Data/OCT/ISMR OCT Data - Downsampled/Depth/"
    dataset = RealDataset(data_root=data_path, labels_dir="all")
    for idx in range(len(dataset)):
        volume, _ = dataset[idx]

        volume_path = dataset.get_volume_path(idx)
        volume_path = volume_path.replace(data_path, depth_path)
        volume_path = volume_path.replace(".npy", ".png")

        volume = adjust_black_white_levels(volume * 255.0, 130, 160) / 255.0
        depthmap = volume_to_depth(volume)
        depthmap = Image.fromarray(depthmap)
        depthmap = depthmap.convert('L')
        depthmap.save(volume_path)
