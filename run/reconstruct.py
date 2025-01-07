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
import os

import torch
from PIL.ImageFile import ImageFile
from tqdm import tqdm

from decalib.datasets import test_data
from decalib.deca import DECA
from decalib.utils.config import cfg as deca_cfg

logger = logging.getLogger(__name__)


class MLRunner:

    def __init__(
        self,
    ):
        self.device = "cuda"

        # run DECA
        deca_cfg.model.use_texture = False
        deca_cfg.model.extract_texture = True
        deca_cfg.rasterizer_type = "pytorch3d"

        self.deca = DECA(config=deca_cfg, device=self.device)

    def run(
        self,
        input_image: ImageFile,
        output_path: str
    ) -> str:
        logger.info(f"Starting generation")
        # load test images
        testdata = test_data.TestData(
            image=input_image,
            face_detector="fan",
        )
        device = 'cuda'

        output_file_path = ""

        for i in tqdm(range(len(testdata))):
            name = testdata[i]["image_name"]
            images = testdata[i]["image"].to(device)[None, ...]

            with torch.no_grad():
                codedict = self.deca.encode(images)
                opdict, visdict = self.deca.decode(codedict)  # tensor

            output_file_path = os.path.join(output_path, f"{name}.obj")
            self.deca.save_obj(output_file_path, opdict)

        logger.info("Finished generation")
        return output_file_path
