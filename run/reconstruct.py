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

import os
import sys

import cv2
import numpy as np
import torch
from scipy.io import savemat
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from decalib.deca import DECA
from decalib.datasets import test_data
from decalib.utils import util
from decalib.utils.config import cfg as deca_cfg


def main(
    input_path: str,
    save_folder: str,
    device: str,
    is_crop: bool,
    detector: str,
    rasterizer_type: str,
    use_texture: bool,
    extract_texture: bool,
    save_keypoints: bool,
    save_depth: bool,
    save_obj: bool,
    save_mat: bool,
):
    save_folder = save_folder
    device = device
    os.makedirs(save_folder, exist_ok=True)

    # load test images 
    testdata = test_data.TestData(
        data_path=input_path,
        is_crop=is_crop,
        face_detector=detector,
    )

    # run DECA
    deca_cfg.model.use_texture = use_texture
    deca_cfg.rasterizer_type = rasterizer_type
    deca_cfg.model.extract_texture = extract_texture
    deca = DECA(config = deca_cfg, device=device)

    for i in tqdm(range(len(testdata))):
        name = testdata[i]['image_name']
        images = testdata[i]['image'].to(device)[None,...]

        with torch.no_grad():
            codedict = deca.encode(images)
            opdict, visdict = deca.decode(codedict) #tensor

        if save_depth or save_keypoints or save_obj or save_mat:
            os.makedirs(os.path.join(save_folder, name), exist_ok=True)

        # save results
        if save_depth:
            depth_image = deca.render.render_depth(opdict['trans_verts']).repeat(1,3,1,1)
            visdict['depth_images'] = depth_image

            cv2.imwrite(
                os.path.join(save_folder, name, name + '_depth.jpg'),
                util.tensor2image(depth_image[0])
            )

        if save_keypoints:
            np.savetxt(
                os.path.join(save_folder, name, name + '_kpt2d.txt'),
                opdict['landmarks2d'][0].cpu().numpy()
            )
            np.savetxt(
                os.path.join(save_folder, name, name + '_kpt3d.txt'),
                opdict['landmarks3d'][0].cpu().numpy()
            )

        if save_obj:
            deca.save_obj(os.path.join(save_folder, name, name + '.obj'), opdict)

        if save_mat:
            opdict = util.dict_tensor2npy(opdict)
            savemat(os.path.join(save_folder, name, name + '.mat'), opdict)

    print(f'-- please check the results in {save_folder}')
