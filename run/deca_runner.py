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
import os
import torch
import trimesh
from PIL.ImageFile import ImageFile
from tqdm import tqdm
from typing import List
import cv2

from ..decalib.datasets import test_data
from ..decalib.deca import DECA
from ..decalib.utils.config import cfg as deca_cfg

logger = logging.getLogger(__name__)


class DecaRunner:

    def __init__(self):
        self.device = "cuda"

        deca_cfg.model.use_texture = False
        deca_cfg.model.extract_texture = False
        deca_cfg.rasterizer_type = "pytorch3d"

        self.deca = DECA(config=deca_cfg, device=self.device)

    def reconstruct(
        self,
        input_images: List[np.ndarray],
        output_file_path: str
    ) -> str:
        logger.info("[DECA Runner] Starting reconstruction")
        # load test images
        dataset = test_data.TestData(images=input_images)

        for i in tqdm(range(len(dataset))):
            image = dataset[i]["image"].to(self.device)[None, ...]

            with torch.no_grad():
                codedict = self.deca.encode(image)
                opdict, visdict = self.deca.decode(codedict)  # tensor

            self.deca.save_obj(output_file_path, opdict)

            vis_path = output_file_path[: output_file_path.rfind(".obj")] + "_vis.jpg"
            cv2.imwrite(
                vis_path,
                self.deca.visualize(visdict),
            )

            # mesh = trimesh.load_mesh(output_file_path + '.obj')
            # mesh.export(output_file_path + '.glb')

        logger.info("[DECA Runner] Finished reconstruction")
        return output_file_path

    def transfer_emotions(
        self,
        input_images: List[np.ndarray],
        emotions_images: List[ImageFile],
        emotions_images_filenames: List[str],
        output_files_path: str,
    ) -> List[str]:
        logger.info("[DECA Runner] Starting transferring emotions")

        input_dataset = test_data.TestData(images=input_images)
        emotions_dataset = test_data.TestData(images=emotions_images)

        output_files_paths = []

        for i in tqdm(range(len(input_dataset))):
            logger.info("[DECA Runner] Starting id reconstruction")
            image = input_dataset[i]["image"].to(self.device)[None, ...]

            with torch.no_grad():
                id_codedict = self.deca.encode(image)
                id_opdict, id_visdict = self.deca.decode(id_codedict)

            id_visdict = { x : id_visdict[x] for x in ["inputs", "shape_detail_images"] }

            for j in tqdm(range(len(emotions_dataset))):
                emotions_file_name = emotions_images_filenames[j]

                logger.info(f"[DECA Runner] Starting transferring emotion from image {emotions_file_name}")
                emotion_image = emotions_dataset[j]["image"].to(self.device)[None, ...]

                with torch.no_grad():
                    emotion_codedict = self.deca.encode(emotion_image)

                id_codedict["pose"][:, 3:] = emotion_codedict["pose"][:, 3:]
                id_codedict["exp"] = emotion_codedict["exp"]

                transfer_opdict, emotion_visdict = self.deca.decode(id_codedict)
                transfer_opdict["uv_texture_gt"] = id_opdict["uv_texture_gt"]

                output_file_path = output_files_path.format(emotions_file_name)

                self.deca.save_obj(output_file_path, transfer_opdict)
                output_files_paths.append(output_file_path)

                id_visdict["transferred_shape"] = emotion_visdict["shape_detail_images"]

                vis_file_path = output_file_path[: output_file_path.rfind(".obj")] + "_vis.jpg"
                cv2.imwrite(
                    vis_file_path,
                    self.deca.visualize(id_visdict),
                )

                logger.info(f"[DECA Runner] Finished transferring emotion from image {emotions_file_name}, file path {output_file_path}")

        logger.info("[DECA Runner] Finished transferring emotions")
        return output_files_paths