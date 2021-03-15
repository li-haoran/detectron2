# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.

from detectron2.config import CfgNode as CN


def add_conl_config(cfg):
    """
    Add config for PointRend.
    """
    # We retry random cropping until no single category in semantic segmentation GT occupies more
    cfg.MODEL.CONL = CN()

    cfg.MODEL.CONL.STAGES=['res4']
    cfg.MODEL.CONL.BLOCKS=[[-1,],]

    cfg.MODEL.CONL.RATIO = 1.0/4.0
    cfg.MODEL.CONL.DOWNSAMPLE=True
    cfg.MODEL.CONL.USE_GN=False
    cfg.MODEL.CONL.LR_MULT=None
    cfg.MODEL.CONL.USE_OUT=False
    cfg.MODEL.CONL.USE_BN=False
    cfg.MODEL.CONL.WHITEN_TYPE=['channel']
    cfg.MODEL.CONL.TEMP = 1.0
    cfg.MODEL.CONL.WITH_GC=False
    cfg.MODEL.CONL.WITH_2FC=False
    cfg.MODEL.CONL.DOUBLE_CONV=False

    cfg.MODEL.CONL.WITH_STATE=False
    cfg.MODEL.CONL.NCLS=32
