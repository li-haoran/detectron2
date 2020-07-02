import torch
import torch.nn as nn
import torch.nn.functional as F 
import fvcore.nn.weight_init as weight_init
import math

from detectron2.layers import ShapeSpec
from pointscollection.layers.points_collection_ops import PointsCollectPack


class PointsCollectionHead(nn.Module):
    def __init__(self,cfg,feature_shape:ShapeSpec):
        super(PointsCollectionHead,self).__init__()

        self.pc=nn.ModuleList()
        self.points_size=1
        for x in feature_shape:
            self.pc.append(PointsCollectPack())
            self.points_size*=9

        self.output_channels=self.points_size*2

    def forward(self, target_offset,dcn_offsets):
        '''
        input:
            [['res3','res4',res5'],'target']

        output:
            target

        '''
        
        
        dcn_offsets=[dcn_offsets[i] for i in range(len(dcn_offsets)-1,-1,-1)]
        for m,dcn_offset in zip(self.pc,dcn_offsets):
            target_offset=m(target_offset,dcn_offset)


        return target_offset

class ClsHead(nn.Module):
    def __init__(self, cfg,  input_shape: ShapeSpec):
        """
        PointsCollectionHead
        """
        super(ClsHead,self).__init__()


        # fmt: off
        in_channels             = input_shape[0].channels
        num_classes             = cfg.MODEL.POINTS_COLLECTION.NUM_CLASSES
        cls_channels            = cfg.MODEL.POINTS_COLLECTION.CLS_CHANNELS
        num_convs               = cfg.MODEL.POINTS_COLLECTION.NUM_CONVS

        # class subnet
        cls_subnet = []
        cur_channels = in_channels
        for _ in range(num_convs):
            cls_subnet.append(
                nn.Conv2d(cur_channels, cls_channels, kernel_size=3, stride=1, padding=1)
            )
            cur_channels = cls_channels
            cls_subnet.append(nn.ReLU())

        self.cls_subnet = nn.Sequential(*cls_subnet)
        self.cls_score = nn.Conv2d(
            cur_channels,  num_classes, kernel_size=3, stride=1, padding=1
        )
        modules_list = [self.cls_subnet, self.cls_score]


        # Initialization
        for modules in modules_list:
            for layer in modules.modules():
                if isinstance(layer, nn.Conv2d):
                    torch.nn.init.normal_(layer.weight, mean=0, std=0.01)
                    torch.nn.init.constant_(layer.bias, 0)

        # Use prior in model initialization to improve stability
        bias_value = -(math.log((1 - 0.01) / 0.01))
        torch.nn.init.constant_(self.cls_score.bias, bias_value)

    def forward(self, features):
        """
        Arguments:
            features (list[Tensor]): FPN feature map tensors in high to low resolution.
                Each tensor in the list correspond to different feature levels.

        Returns:
            pred_logits (list[Tensor]): #lvl tensors, each has shape (N, AxK, Hi, Wi).
                The tensor predicts the classification probability
                at each spatial position for each of the A anchors and K object
                classes.
        """
        x=features[0]

        pred_logits = self.cls_score(self.cls_subnet(x)) 
        return pred_logits
