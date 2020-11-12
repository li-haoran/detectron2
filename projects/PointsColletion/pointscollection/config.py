# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from detectron2.config import CfgNode as CN


def add_pointscollection_config(cfg):
    """
    Add config for TensorMask.
    """
    cfg.MODEL.POINTS_COLLECTION = CN()

    # Anchor parameters
    cfg.MODEL.POINTS_COLLECTION.MASK_ON = False

    cfg.MODEL.POINTS_COLLECTION.NUM_CLASSES = 80
    cfg.MODEL.POINTS_COLLECTION.CLS_CHANNELS = 256
    cfg.MODEL.POINTS_COLLECTION.NUM_CONVS = 2
    cfg.MODEL.POINTS_COLLECTION.CIN_FEATURES = ["res5"]
    cfg.MODEL.POINTS_COLLECTION.PIN_FEATURES = ["res3","res4","res5"]
    cfg.MODEL.POINTS_COLLECTION.MIN_FEATURES = ["res3"]

    cfg.MODEL.POINTS_COLLECTION.INS_CHANNELS = 256
    cfg.MODEL.POINTS_COLLECTION.INS_NUM_CONVS = 2


    # Loss parameters
    cfg.MODEL.POINTS_COLLECTION.FOCAL_LOSS_GAMMA = 2.0
    cfg.MODEL.POINTS_COLLECTION.FOCAL_LOSS_ALPHA = 0.25

    cfg.MODEL.POINTS_COLLECTION.MASK_LOSS_WEIGHT = 1.0
    cfg.MODEL.POINTS_COLLECTION.INS_LOSS_WEIGHT = 1.0
    cfg.MODEL.POINTS_COLLECTION.CIRCUM = False

    cfg.MODEL.POINTS_COLLECTION.SCORE_THRESH_TEST =0.1

    cfg.MODEL.POINTS_COLLECTION.SIGMA = 2
    cfg.MODEL.POINTS_COLLECTION.CONTOUR = 729


    cfg.INPUT.MASK_FORMAT = "bitmask" 



    cfg.MODEL.VGG = CN()

    cfg.MODEL.VGG.DEPTH = 16
    cfg.MODEL.VGG.OUT_FEATURES = ["res3","res4","res5"]

    # Options: FrozenBN, GN, "SyncBN", "BN"
    cfg.MODEL.VGG.NORM = "FrozenBN"

    # Output channels of conv5 block
    cfg.MODEL.VGG.CONV5_OUT_CHANNELS = 512
    cfg.MODEL.VGG.DEFORM_ON_PER_STAGE=[False,False,True,True,True]