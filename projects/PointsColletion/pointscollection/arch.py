# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import copy
import logging
import math
from typing import List
import torch
import torch.nn.functional as F
from fvcore.nn import sigmoid_focal_loss, smooth_l1_loss
from torch import nn
import numpy as np

from detectron2.layers import ShapeSpec, cat
from detectron2.modeling.backbone import build_backbone
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY

from detectron2.structures import Boxes, ImageList, Instances
from detectron2.utils.logger import log_first_n

from pointscollection.head import PointsCollectionHead,ClsHead
from pointscollection.data import Targets,_postprocess



@META_ARCH_REGISTRY.register()
class PointsCollection(nn.Module):
    """
    TensorMask model. Creates FPN backbone, anchors and a head for classification
    and box regression. Calculates and applies proper losses to class, box, and
    masks.
    """

    def __init__(self, cfg):
        super().__init__()

        # fmt: off
        self.num_classes              = cfg.MODEL.POINTS_COLLECTION.NUM_CLASSES
        self.cin_features              = cfg.MODEL.POINTS_COLLECTION.CIN_FEATURES
        self.pin_features              =cfg.MODEL.POINTS_COLLECTION.PIN_FEATURES
        # Loss parameters:
        self.focal_loss_alpha         = cfg.MODEL.POINTS_COLLECTION.FOCAL_LOSS_ALPHA
        self.focal_loss_gamma         = cfg.MODEL.POINTS_COLLECTION.FOCAL_LOSS_GAMMA

        self.mask_loss_weight         =cfg.MODEL.POINTS_COLLECTION.MASK_LOSS_WEIGHT
        self.circumscribed            =cfg.MODEL.POINTS_COLLECTION.CIRCUM

        self.score_threshold          = cfg.MODEL.POINTS_COLLECTION.SCORE_THRESH_TEST

        # build the backbone
        self.backbone = build_backbone(cfg)

        backbone_shape = self.backbone.output_shape()
        classify_feature_shapes = [backbone_shape[f] for f in self.cin_features]
        self.classify_feature_strides = [x.stride for x in classify_feature_shapes]

        assert len(classify_feature_shapes)==1,'here just use final output'
        points_feature_shapes=[backbone_shape[f] for f in self.pin_features]
        self.points_feature_strides=[x.stride for x in points_feature_shapes]

        self.pc_head=PointsCollectionHead(cfg,points_feature_shapes)
        self.cls_head=ClsHead(cfg,classify_feature_shapes)

        self.target_generator=Targets(cfg,self.points_feature_strides[-1],self.classify_feature_strides[-1])
        self.register_buffer("pixel_mean", torch.Tensor(cfg.MODEL.PIXEL_MEAN).view(-1, 1, 1))
        self.register_buffer("pixel_std", torch.Tensor(cfg.MODEL.PIXEL_STD).view(-1, 1, 1))

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DetectionTransform` .
                Each item in the list contains the inputs for one image.
            For now, each item in the list is a dict that contains:
                image: Tensor, image in (C, H, W) format.
                instances: Instances
                Other information that's included in the original dicts, such as:
                    "height", "width" (int): the output resolution of the model, used in inference.
                        See :meth:`postprocess` for details.
         Returns:
            losses (dict[str: Tensor]): mapping from a named loss to a tensor
                storing the loss. Used during training only.
        """
        images = self.preprocess_image(batched_inputs)
        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        elif "targets" in batched_inputs[0]:
            log_first_n(
                logging.WARN, "'targets' in the model inputs is now renamed to 'instances'!", n=10
            )
            gt_instances = [x["targets"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        # print(images.image_sizes)
        # print(images.tensor.size())
        features = self.backbone(images.tensor)
        classify_features = [features[f][0] for f in self.cin_features]
        points_features =[features[f][1] for f in self.pin_features]
        # apply the head
        # print(classify_features[0].size())

        pf_b,pf_c,pf_h,pf_w=points_features[-1].size()
        target_points=points_features[-1].new_zeros(pf_b,2,pf_h,pf_w,requires_grad=False)
        pred_digits=self.cls_head(classify_features)
        pred_points=self.pc_head(target_points,points_features)

        if self.training:
            # get ground truths for class labels and box targets, it will label each anchor
            output_size=classify_features[-1].size()
            gt_clses, gt_belongs, gt_masks, = self.get_ground_truth(gt_instances,output_size)
            # compute the loss
            return self.losses(
                gt_clses,
                gt_belongs,
                gt_masks,
                pred_digits,
                pred_points,
            )
        else:
            # do inference to get the output
            results = self.inference(pred_digits, pred_points,images)
            processed_results = []
            for results_im, input_im, image_size in zip(
                results, batched_inputs, images.image_sizes
            ):
                height = input_im.get("height", image_size[0])
                width = input_im.get("width", image_size[1])
                # this is to do post-processing with the image size
                result= results_im
                r = _postprocess(result,height, width)
                processed_results.append({"instances": r})
            return processed_results

    def losses(
        self,
        gt_clses,
        gt_belongs,
        gt_masks,
        pred_logits,
        pred_points,
    ):
        """
        Args:
            For `gt_clses`, `gt_belongs`, `gt_masks` parameters, see
                :meth:`TensorMask.get_ground_truth`.
            For `pred_logits` and `pred_points`, see
                :meth:`TensorMaskHead.forward`.

        Returns:
            losses (dict[str: Tensor]): mapping from a named loss to a scalar tensor
                storing the loss. Used during training only. The potential dict keys are:
                "loss_cls", and "loss_mask".
        """
        num_fg=gt_belongs.size(0)
        # print(num_fg,torch.sum(gt_clses))
        loss_normalizer = torch.tensor(max(1, num_fg), dtype=torch.float32, device=self.device)

        # classification
        loss_cls = (
            sigmoid_focal_loss(
                pred_logits,
                gt_clses,
                alpha=self.focal_loss_alpha,
                gamma=self.focal_loss_gamma,
                reduction="sum",
            )
            / loss_normalizer
        )


        losses = {"loss_cls": loss_cls, }

        # mask prediction
        loss_mask = 0

        pred_points_valids=pred_points[gt_belongs[:,0],:,gt_belongs[:,1],gt_belongs[:,2]]

        N,P=pred_points_valids.size()
        pred_points_valids=pred_points_valids.view(N,P//2,2,1)

        Q=gt_masks.size(1)
        gt_masks=gt_masks.view(N,1,Q,2)
        gt_masks=gt_masks.transpose(2,3)

        pred_points_valids_contour=pred_points_valids

        if self.circumscribed:
            num_points=P//2
            edge=np.sqrt(num_points)       
            # top=[0+i for i in range(edge)]
            # left=[i*edge+0 for i in range(1,edge-1)]
            # right=[i*edge+edge-1 for i in range(1,edge-1)]
            # bottom=[(edge-1)*edge+i for i in range(edge)]
            # Index=top+left+right+bottom
            All=[i*edge+j for i in range(edge) for j in range(edge)]
            Inner=[i*edge+j for i in range(1,edge-1) for j in range(1,edge-1)]
            Out=list(set(All)-set(Inner))
            Out=torch.tensor(Out,dtype=torch.int64,device=self.device)
            pred_points_valids_contour=pred_points_valids[:,Out,:,:]

            
            # Inner=torch.tensor(Inner,dtype=torch.int64,device=self.device)
            # pred_points_valids_inner=pred_points_valids[:,Inner,:,:]
        
        l1_loss = torch.abs(pred_points_valids_contour - gt_masks)
        distance = torch.sum(l1_loss,dim=2)
        min_l1_loss,_=torch.min(distance,dim=2)

        loss_mask=torch.mean(min_l1_loss)*self.mask_loss_weight
        
        losses["loss_mask"] = loss_mask
        return losses

    @torch.no_grad()
    def get_ground_truth(self, targets,output_size):
        """
        Args:

            targets (list[Instances]): a list of N `Instances`s. The i-th
                `Instances` contains the ground-truth per-instance annotations
                for the i-th input image.  Specify `targets` during training only.
            output_size: the output featuremap size.

        Returns:
            gt_clses:   b x c x h x w tensor
            gt_belongs: n x 3(b,y,x) tensor
            gt_masks: n x m(81*2) tensor
        """
        gt_clses=[]
        gt_belongs=[]
        gt_masks=[]
        for it,target_im in enumerate(targets):
            classes=target_im.gt_classes
            masks=target_im.gt_masks       
            gt_cls,gt_belong,gt_mask=self.target_generator.get_target_single(classes,masks,output_size)

            batch_index=np.zeros((gt_belong.shape[0],1),dtype=np.int64)+it
            gt_belong_batch=np.concatenate([batch_index,gt_belong],axis=1)

            gt_clses.append(gt_cls)
            gt_belongs.append(gt_belong_batch)
            gt_masks.append(gt_mask)

        gt_clses=np.stack(gt_clses)
        gt_clses=torch.from_numpy(gt_clses).to(device=self.device)

        gt_clses_binary=torch.tensor(gt_clses>0,dtype=torch.float32,device=self.device)
        
        gt_belongs=np.concatenate(gt_belongs)
        gt_belongs=torch.from_numpy(gt_belongs).to(device=self.device)
        gt_masks=np.concatenate(gt_masks)
        gt_masks=torch.from_numpy(gt_masks).to(device=self.device)

        return [gt_clses_binary,gt_belongs,gt_masks]
       

    def inference(self, pred_digits,pred_points,images):
        """
        Arguments:
            pred_digits,  pred_points: Same as the output of:
            
            images (ImageList): the input images

        Returns:
            results (List[Instances]): a list of #images elements.
        """
        batch=pred_digits.size(0)
        pred_digits=pred_digits.sigmoid_()
        results=[]

        pool_digits=F.max_pool2d(pred_digits,3,1,1)

        for img_idx in range(batch):
            # Get the size of the current image
            image_size = images.image_sizes[img_idx]

            digits_im = pred_digits[img_idx]
            pool_digits_im=pred_digits[img_idx]
            points_im=pred_points[img_idx]

            Index=torch.nonzero((digits_im==pool_digits_im) & (digits_im>self.score_threshold))

            results_im=Instances(image_size)
            if Index.size(0)<1:
                continue

            cls_idxs=Index[:,0]
            pred_prob=digits_im[Index[:,0],Index[:,1],Index[:,2]]

            center=torch.cat([Index[:,2:3],Index[:,1:2]],dim=1)

            N=center.size(0)
            center=center.view(N,1,2)
            
            points_n = points_im[:,Index[:,1],Index[:,2]]
            npoints=torch.transpose(points_n)
            npoints=npoints.view(N,-1,2)

            real_npoints=npoints+center

            real_npoints=real_npoints*self.points_feature_strides[-1]

            top_left=torch.min(real_npoints,dim=1)
            bottom_right=torch.max(real_npoints,dim=1)

            bbox=torch.cat([top_left,bottom_right],dim=1)

            results_im.pred_classes = cls_idxs
            results.pred_boxes = Boxes(bbox)
            results.scores = pred_prob
            results.pred_points=real_npoints          
            results.append(results_im)
        return results

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images


