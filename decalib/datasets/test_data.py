# -*- coding: utf-8 -*-
#
# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# Using this computer program means that you agree to the terms 
# in the LICENSE file included with this software distribution. 
# Any use not explicitly granted by the LICENSE is prohibited.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# For comments or questions, please email us at deca@tue.mpg.de
# For commercial licensing contact, please contact ps-license@tuebingen.mpg.de
import logging

import numpy as np
import torch
from PIL.ImageFile import ImageFile
from skimage.transform import estimate_transform, warp
from torch.utils.data import Dataset

from .detectors import FAN

logger = logging.getLogger(__name__)


class TestData(Dataset):

    def __init__(
        self, 
        image: ImageFile,
        crop_size: int = 224,
        scale: float = 1.25,
        face_detector: str = "fan",
    ):
        self.images_list = [image]
        self.crop_size = crop_size
        self.scale = scale
        self.resolution_inp = crop_size

        if face_detector == "fan":
            self.face_detector = FAN()
        # elif face_detector == "mtcnn":
        #     self.face_detector = detectors.MTCNN()
        else:
            print(f"please check the detector: {face_detector}")
            exit()

    def __len__(self):
        return len(self.images_list)

    def bbox2point(self, left, right, top, bottom, type="bbox"):
        """
        bbox from detector and landmarks are different
        """
        if type == "kpt68":
            old_size = (right - left + bottom - top)/2*1.1
            center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0 ])
        elif type=="bbox":
            old_size = (right - left + bottom - top)/2
            center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0  + old_size*0.12])
        else:
            raise NotImplementedError
        return old_size, center

    def __getitem__(self, index):
        image = self.images_list[index]
        image_name = image.filename

        image = np.array(image)

        if len(image.shape) == 2:
            image = image[:,:,None].repeat(1,1,3)
        if len(image.shape) == 3 and image.shape[2] > 3:
            image = image[:,:,:3]

        h, w, _ = image.shape

        bbox, bbox_type = self.face_detector.run(image)
        if len(bbox) < 4:
            logger.info("No face detected! run original image")
            left = 0
            right = h - 1
            top = 0
            bottom = w - 1
        else:
            left = bbox[0]
            right = bbox[2]
            top = bbox[1];
            bottom = bbox[3]
        old_size, center = self.bbox2point(left, right, top, bottom, type=bbox_type)

        size = int(old_size * self.scale)
        src_pts = np.array(
            [
                [center[0] - size / 2, center[1] - size / 2],
                [center[0] - size / 2, center[1] + size / 2],
                [center[0] + size / 2, center[1] - size / 2]
            ]
        )

        DST_PTS = np.array(
            [
                [0,0],
                [0, self.resolution_inp - 1],
                [self.resolution_inp - 1, 0]
            ]
        )
        tform = estimate_transform("similarity", src_pts, DST_PTS)
        
        image = image/255

        dst_image = warp(image, tform.inverse, output_shape=(self.resolution_inp, self.resolution_inp))
        dst_image = dst_image.transpose(2,0,1)

        return {
            "image": torch.tensor(dst_image).float(),
            "image_name": image_name,
            "tform": torch.tensor(tform.params).float(),
            "original_image": torch.tensor(image.transpose(2,0,1)).float(),
        }
