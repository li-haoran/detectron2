# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from .config import add_pointscollection_config
from .arch import PointsCollection
from .arch_ins import PointsCollectionIns
from .arch_refine import PointsCollectionRefine
from .backbone import build_points_collection_resnet_backbone,ResNet2,DeformbleOffsetBottleneckBlock
from .backbone2 import build_points_collection_vgg_backbone,VGG,DeformConv2
