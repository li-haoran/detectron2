# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from .config import add_pointscollection_config
from .arch import PointsCollection
from .backbone import build_points_collection_resnet_backbone,ResNet2,DeformbleOffsetBottleneckBlock