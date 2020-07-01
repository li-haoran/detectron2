# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from detectron2.config import CfgNode as CN


def add_tensormask_config(cfg):
    """
    Add config for TensorMask.
    """
    cfg.MODEL.POINTS_COLLECTION = CN()

    # Anchor parameters
    cfg.MODEL.POINTS_COLLECTION.NUM_CLASSES = 80
    cfg.MODEL.POINTS_COLLECTION.CIN_FEATURES = ["res5"]
    cfg.MODEL.POINTS_COLLECTION.PIN_FEATURES = ["res3","res4","res5"]

    # Loss parameters
    cfg.MODEL.POINTS_COLLECTION.FOCAL_LOSS_GAMMA = 3.0
    cfg.MODEL.POINTS_COLLECTION.FOCAL_LOSS_ALPHA = 0.3

    cfg.MODEL.POINTS_COLLECTION.MASK_LOSS_WEIGHT = 1.0
    cfg.MODEL.POINTS_COLLECTION.CIRCUM = False

    cfg.MODEL.POINTS_COLLECTION.SCORE_THRESH_TEST =0.05

    cfg.MODEL.POINTS_COLLECTION.SIGMA = 2
    cfg.MODEL.POINTS_COLLECTION.CONTOUR = 81


    cfg.INPUT.MASK_FORMAT = "bitmask" 