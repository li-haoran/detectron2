# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from detectron2.config import CfgNode as CN


def add_tensormask_config(cfg):
    """
    Add config for TensorMask.
    """
    cfg.MODEL.POINTS_COLLECTION = CN()

    # Anchor parameters
    cfg.MODEL.POINTS_COLLECTION.IN_FEATURES = ["p5"]

