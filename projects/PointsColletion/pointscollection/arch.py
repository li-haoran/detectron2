# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import copy
import logging
import math
from typing import List
import torch
import torch.nn.functional as F
from fvcore.nn import sigmoid_focal_loss_star_jit, smooth_l1_loss
from torch import nn

from detectron2.layers import ShapeSpec, batched_nms, cat, paste_masks_in_image
from detectron2.modeling.anchor_generator import DefaultAnchorGenerator
from detectron2.modeling.backbone import build_backbone
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.modeling.meta_arch.retinanet import (
    permute_all_cls_and_box_to_N_HWA_K_and_concat,
    permute_to_N_HWA_K,
)
from detectron2.structures import Boxes, ImageList, Instances
from detectron2.utils.logger import log_first_n


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
        self.in_features              = cfg.MODEL.POINTS_COLLECTION.IN_FEATURES
        
        # # Loss parameters:
        # self.focal_loss_alpha         = cfg.MODEL.POINTS_COLLECTION.FOCAL_LOSS_ALPHA
        # self.focal_loss_gamma         = cfg.MODEL.POINTS_COLLECTION.FOCAL_LOSS_GAMMA
        # # Inference parameters:
        # self.score_threshold          = cfg.MODEL.POINTS_COLLECTION.SCORE_THRESH_TEST
        # self.topk_candidates          = cfg.MODEL.POINTS_COLLECTION.TOPK_CANDIDATES_TEST
        # self.nms_threshold            = cfg.MODEL.POINTS_COLLECTION.NMS_THRESH_TEST
        # self.detections_im            = cfg.TEST.DETECTIONS_PER_IMAGE
        # # Mask parameters:
        # self.mask_on                  = cfg.MODEL.MASK_ON
        # self.mask_loss_weight         = cfg.MODEL.POINTS_COLLECTION.MASK_LOSS_WEIGHT
        # self.mask_pos_weight          = torch.tensor(cfg.MODEL.POINTS_COLLECTION.POSITIVE_WEIGHT,
        #                                              dtype=torch.float32)
        # self.bipyramid_on             = cfg.MODEL.POINTS_COLLECTION.BIPYRAMID_ON
        # # fmt: on

        # build the backbone
        self.backbone = build_backbone(cfg)

        backbone_shape = self.backbone.output_shape()
        feature_shapes = [backbone_shape[f] for f in self.in_features]
        feature_strides = [x.stride for x in feature_shapes]
       

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

        features = self.backbone(images.tensor)
        features = [features[f] for f in self.in_features]
        # apply the TensorMask head
        pred_logits, pred_deltas, pred_masks = self.head(features)
        # generate anchors based on features, is it image specific?
        anchors, unit_lengths, indexes = self.anchor_generator(features)

        if self.training:
            # get ground truths for class labels and box targets, it will label each anchor
            gt_class_info, gt_delta_info, gt_mask_info, num_fg = self.get_ground_truth(
                anchors, unit_lengths, indexes, gt_instances
            )
            # compute the loss
            return self.losses(
                gt_class_info,
                gt_delta_info,
                gt_mask_info,
                num_fg,
                pred_logits,
                pred_deltas,
                pred_masks,
            )
        else:
            # do inference to get the output
            results = self.inference(pred_logits, pred_deltas, pred_masks, anchors, indexes, images)
            processed_results = []
            for results_im, input_im, image_size in zip(
                results, batched_inputs, images.image_sizes
            ):
                height = input_im.get("height", image_size[0])
                width = input_im.get("width", image_size[1])
                # this is to do post-processing with the image size
                result_box, result_mask = results_im
                r = _postprocess(result_box, result_mask, height, width)
                processed_results.append({"instances": r})
            return processed_results

    def losses(
        self,
        gt_class_info,
        gt_delta_info,
        gt_mask_info,
        num_fg,
        pred_logits,
        pred_deltas,
        pred_masks,
    ):
        """
        Args:
            For `gt_class_info`, `gt_delta_info`, `gt_mask_info` and `num_fg` parameters, see
                :meth:`TensorMask.get_ground_truth`.
            For `pred_logits`, `pred_deltas` and `pred_masks`, see
                :meth:`TensorMaskHead.forward`.

        Returns:
            losses (dict[str: Tensor]): mapping from a named loss to a scalar tensor
                storing the loss. Used during training only. The potential dict keys are:
                "loss_cls", "loss_box_reg" and "loss_mask".
        """
        gt_classes_target, gt_valid_inds = gt_class_info
        gt_deltas, gt_fg_inds = gt_delta_info
        gt_masks, gt_mask_inds = gt_mask_info
        loss_normalizer = torch.tensor(max(1, num_fg), dtype=torch.float32, device=self.device)

        # classification and regression
        pred_logits, pred_deltas = permute_all_cls_and_box_to_N_HWA_K_and_concat(
            pred_logits, pred_deltas, self.num_classes
        )
        loss_cls = (
            sigmoid_focal_loss_star_jit(
                pred_logits[gt_valid_inds],
                gt_classes_target[gt_valid_inds],
                alpha=self.focal_loss_alpha,
                gamma=self.focal_loss_gamma,
                reduction="sum",
            )
            / loss_normalizer
        )

        if num_fg == 0:
            loss_box_reg = pred_deltas.sum() * 0
        else:
            loss_box_reg = (
                smooth_l1_loss(pred_deltas[gt_fg_inds], gt_deltas, beta=0.0, reduction="sum")
                / loss_normalizer
            )
        losses = {"loss_cls": loss_cls, "loss_box_reg": loss_box_reg}

        # mask prediction
        if self.mask_on:
            loss_mask = 0
            for lvl in range(self.num_levels):
                cur_level_factor = 2 ** lvl if self.bipyramid_on else 1
                for anc in range(self.num_anchors):
                    cur_gt_mask_inds = gt_mask_inds[lvl][anc]
                    if cur_gt_mask_inds is None:
                        loss_mask += pred_masks[lvl][anc][0, 0, 0, 0] * 0
                    else:
                        cur_mask_size = self.mask_sizes[anc] * cur_level_factor
                        # TODO maybe there are numerical issues when mask sizes are large
                        cur_size_divider = torch.tensor(
                            self.mask_loss_weight / (cur_mask_size ** 2),
                            dtype=torch.float32,
                            device=self.device,
                        )

                        cur_pred_masks = pred_masks[lvl][anc][
                            cur_gt_mask_inds[:, 0],  # N
                            :,  # V x U
                            cur_gt_mask_inds[:, 1],  # H
                            cur_gt_mask_inds[:, 2],  # W
                        ]

                        loss_mask += F.binary_cross_entropy_with_logits(
                            cur_pred_masks.view(-1, cur_mask_size, cur_mask_size),  # V, U
                            gt_masks[lvl][anc].to(dtype=torch.float32),
                            reduction="sum",
                            weight=cur_size_divider,
                            pos_weight=self.mask_pos_weight,
                        )
            losses["loss_mask"] = loss_mask / loss_normalizer
        return losses

    @torch.no_grad()
    def get_ground_truth(self, anchors, unit_lengths, indexes, targets):
        """
        Args:
            anchors (list[list[Boxes]]): a list of N=#image elements. Each is a
                list of #feature level Boxes. The Boxes contains anchors of
                this image on the specific feature level.
            unit_lengths (list[list[Tensor]]): a list of N=#image elements. Each is a
                list of #feature level Tensor. The tensor contains unit lengths for anchors of
                this image on the specific feature level.
            indexes (list[list[Tensor]]): a list of N=#image elements. Each is a
                list of #feature level Tensor. The tensor contains the 5D index of
                each anchor, the second dimension means (L, N, H, W, A), where L
                is level, I is image, H is height, W is width, and A is anchor.
            targets (list[Instances]): a list of N `Instances`s. The i-th
                `Instances` contains the ground-truth per-instance annotations
                for the i-th input image.  Specify `targets` during training only.

        Returns:
            gt_class_info (Tensor, Tensor): A pair of two tensors for classification.
                The first one is an integer tensor of shape (R, #classes) storing ground-truth
                labels for each anchor. R is the total number of anchors in the batch.
                The second one is an integer tensor of shape (R,), to indicate which
                anchors are valid for loss computation, which anchors are not.
            gt_delta_info (Tensor, Tensor): A pair of two tensors for boxes.
                The first one, of shape (F, 4). F=#foreground anchors.
                The last dimension represents ground-truth box2box transform
                targets (dx, dy, dw, dh) that map each anchor to its matched ground-truth box.
                Only foreground anchors have values in this tensor. Could be `None` if F=0.
                The second one, of shape (R,), is an integer tensor indicating which anchors
                are foreground ones used for box regression. Could be `None` if F=0.
            gt_mask_info (list[list[Tensor]], list[list[Tensor]]): A pair of two lists for masks.
                The first one is a list of P=#feature level elements. Each is a
                list of A=#anchor tensors. Each tensor contains the ground truth
                masks of the same size and for the same feature level. Could be `None`.
                The second one is a list of P=#feature level elements. Each is a
                list of A=#anchor tensors. Each tensor contains the location of the ground truth
                masks of the same size and for the same feature level. The second dimension means
                (N, H, W), where N is image, H is height, and W is width. Could be `None`.
            num_fg (int): F=#foreground anchors, used later for loss normalization.
        """
        gt_classes = []
        gt_deltas = []
        gt_masks = [[[] for _ in range(self.num_anchors)] for _ in range(self.num_levels)]
        gt_mask_inds = [[[] for _ in range(self.num_anchors)] for _ in range(self.num_levels)]

        anchors = [Boxes.cat(anchors_i) for anchors_i in anchors]
        unit_lengths = [cat(unit_lengths_i) for unit_lengths_i in unit_lengths]
        indexes = [cat(indexes_i) for indexes_i in indexes]

        num_fg = 0
        for i, (anchors_im, unit_lengths_im, indexes_im, targets_im) in enumerate(
            zip(anchors, unit_lengths, indexes, targets)
        ):
            # Initialize all
            gt_classes_i = torch.full_like(
                unit_lengths_im, self.num_classes, dtype=torch.int64, device=self.device
            )
            # Ground truth classes
            has_gt = len(targets_im) > 0
            if has_gt:
                # Compute the pairwise matrix
                gt_matched_inds, anchor_labels = _assignment_rule(
                    targets_im.gt_boxes, anchors_im, unit_lengths_im, self.min_anchor_size
                )
                # Find the foreground instances
                fg_inds = anchor_labels == 1
                fg_anchors = anchors_im[fg_inds]
                num_fg += len(fg_anchors)
                # Find the ground truths for foreground instances
                gt_fg_matched_inds = gt_matched_inds[fg_inds]
                # Assign labels for foreground instances
                gt_classes_i[fg_inds] = targets_im.gt_classes[gt_fg_matched_inds]
                # Anchors with label -1 are ignored, others are left as negative
                gt_classes_i[anchor_labels == -1] = -1

                # Boxes
                # Ground truth box regression, only for foregrounds
                matched_gt_boxes = targets_im[gt_fg_matched_inds].gt_boxes
                # Compute box regression offsets for foregrounds only
                gt_deltas_i = self.box2box_transform.get_deltas(
                    fg_anchors.tensor, matched_gt_boxes.tensor
                )
                gt_deltas.append(gt_deltas_i)

                # Masks
                if self.mask_on:
                    # Compute masks for each level and each anchor
                    matched_indexes = indexes_im[fg_inds, :]
                    for lvl in range(self.num_levels):
                        ids_lvl = matched_indexes[:, 0] == lvl
                        if torch.any(ids_lvl):
                            cur_level_factor = 2 ** lvl if self.bipyramid_on else 1
                            for anc in range(self.num_anchors):
                                ids_lvl_anchor = ids_lvl & (matched_indexes[:, 4] == anc)
                                if torch.any(ids_lvl_anchor):
                                    gt_masks[lvl][anc].append(
                                        targets_im[
                                            gt_fg_matched_inds[ids_lvl_anchor]
                                        ].gt_masks.crop_and_resize(
                                            fg_anchors[ids_lvl_anchor].tensor,
                                            self.mask_sizes[anc] * cur_level_factor,
                                        )
                                    )
                                    # Select (N, H, W) dimensions
                                    gt_mask_inds_lvl_anc = matched_indexes[ids_lvl_anchor, 1:4]
                                    # Set the image index to the current image
                                    gt_mask_inds_lvl_anc[:, 0] = i
                                    gt_mask_inds[lvl][anc].append(gt_mask_inds_lvl_anc)
            gt_classes.append(gt_classes_i)

        # Classes and boxes
        gt_classes = cat(gt_classes)
        gt_valid_inds = gt_classes >= 0
        gt_fg_inds = gt_valid_inds & (gt_classes < self.num_classes)
        gt_classes_target = torch.zeros(
            (gt_classes.shape[0], self.num_classes), dtype=torch.float32, device=self.device
        )
        gt_classes_target[gt_fg_inds, gt_classes[gt_fg_inds]] = 1
        gt_deltas = cat(gt_deltas) if gt_deltas else None

        # Masks
        gt_masks = [[cat(mla) if mla else None for mla in ml] for ml in gt_masks]
        gt_mask_inds = [[cat(ila) if ila else None for ila in il] for il in gt_mask_inds]
        return (
            (gt_classes_target, gt_valid_inds),
            (gt_deltas, gt_fg_inds),
            (gt_masks, gt_mask_inds),
            num_fg,
        )

    def inference(self, pred_logits, pred_deltas, pred_masks, anchors, indexes, images):
        """
        Arguments:
            pred_logits, pred_deltas, pred_masks: Same as the output of:
                meth:`TensorMaskHead.forward`
            anchors, indexes: Same as the input of meth:`TensorMask.get_ground_truth`
            images (ImageList): the input images

        Returns:
            results (List[Instances]): a list of #images elements.
        """
        assert len(anchors) == len(images)
        results = []

        pred_logits = [permute_to_N_HWA_K(x, self.num_classes) for x in pred_logits]
        pred_deltas = [permute_to_N_HWA_K(x, 4) for x in pred_deltas]

        pred_logits = cat(pred_logits, dim=1)
        pred_deltas = cat(pred_deltas, dim=1)

        for img_idx, (anchors_im, indexes_im) in enumerate(zip(anchors, indexes)):
            # Get the size of the current image
            image_size = images.image_sizes[img_idx]

            logits_im = pred_logits[img_idx]
            deltas_im = pred_deltas[img_idx]

            if self.mask_on:
                masks_im = [[mla[img_idx] for mla in ml] for ml in pred_masks]
            else:
                masks_im = [None] * self.num_levels
            results_im = self.inference_single_image(
                logits_im,
                deltas_im,
                masks_im,
                Boxes.cat(anchors_im),
                cat(indexes_im),
                tuple(image_size),
            )
            results.append(results_im)
        return results

    def inference_single_image(
        self, pred_logits, pred_deltas, pred_masks, anchors, indexes, image_size
    ):
        """
        Single-image inference. Return bounding-box detection results by thresholding
        on scores and applying non-maximum suppression (NMS).

        Arguments:
            pred_logits (list[Tensor]): list of #feature levels. Each entry contains
                tensor of size (AxHxW, K)
            pred_deltas (list[Tensor]): Same shape as 'pred_logits' except that K becomes 4.
            pred_masks (list[list[Tensor]]): List of #feature levels, each is a list of #anchors.
                Each entry contains tensor of size (M_i*M_i, H, W). `None` if mask_on=False.
            anchors (list[Boxes]): list of #feature levels. Each entry contains
                a Boxes object, which contains all the anchors for that
                image in that feature level.
            image_size (tuple(H, W)): a tuple of the image height and width.

        Returns:
            Same as `inference`, but for only one image.
        """
        pred_logits = pred_logits.flatten().sigmoid_()
        # We get top locations across all levels to accelerate the inference speed,
        # which does not seem to affect the accuracy.
        # First select values above the threshold
        logits_top_idxs = torch.where(pred_logits > self.score_threshold)[0]
        # Then get the top values
        num_topk = min(self.topk_candidates, logits_top_idxs.shape[0])
        pred_prob, topk_idxs = pred_logits[logits_top_idxs].sort(descending=True)
        # Keep top k scoring values
        pred_prob = pred_prob[:num_topk]
        # Keep top k values
        top_idxs = logits_top_idxs[topk_idxs[:num_topk]]

        # class index
        cls_idxs = top_idxs % self.num_classes
        # HWA index
        top_idxs //= self.num_classes
        # predict boxes
        pred_boxes = self.box2box_transform.apply_deltas(
            pred_deltas[top_idxs], anchors[top_idxs].tensor
        )
        # apply nms
        keep = batched_nms(pred_boxes, pred_prob, cls_idxs, self.nms_threshold)
        # pick the top ones
        keep = keep[: self.detections_im]

        results = Instances(image_size)
        results.pred_boxes = Boxes(pred_boxes[keep])
        results.scores = pred_prob[keep]
        results.pred_classes = cls_idxs[keep]

        # deal with masks
        result_masks, result_anchors = [], None
        if self.mask_on:
            # index and anchors, useful for masks
            top_indexes = indexes[top_idxs]
            top_anchors = anchors[top_idxs]
            result_indexes = top_indexes[keep]
            result_anchors = top_anchors[keep]
            # Get masks and do sigmoid
            for lvl, _, h, w, anc in result_indexes.tolist():
                cur_size = self.mask_sizes[anc] * (2 ** lvl if self.bipyramid_on else 1)
                result_masks.append(
                    torch.sigmoid(pred_masks[lvl][anc][:, h, w].view(1, cur_size, cur_size))
                )

        return results, (result_masks, result_anchors)

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images


class TensorMaskHead(nn.Module):
    def __init__(self, cfg, num_levels, num_anchors, mask_sizes, input_shape: List[ShapeSpec]):
        """
        TensorMask head.
        """
        super().__init__()
        # fmt: off
        self.in_features        = cfg.MODEL.POINTS_COLLECTION.IN_FEATURES
        in_channels             = input_shape[0].channels
        num_classes             = cfg.MODEL.POINTS_COLLECTION.NUM_CLASSES
        cls_channels            = cfg.MODEL.POINTS_COLLECTION.CLS_CHANNELS
        num_convs               = cfg.MODEL.POINTS_COLLECTION.NUM_CONVS
        # box parameters
        bbox_channels           = cfg.MODEL.POINTS_COLLECTION.BBOX_CHANNELS
        # mask parameters
        self.mask_on            = cfg.MODEL.MASK_ON
        self.mask_sizes         = mask_sizes
        mask_channels           = cfg.MODEL.POINTS_COLLECTION.MASK_CHANNELS
        self.align_on           = cfg.MODEL.POINTS_COLLECTION.ALIGNED_ON
        self.bipyramid_on       = cfg.MODEL.POINTS_COLLECTION.BIPYRAMID_ON
        # fmt: on

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
            cur_channels, num_anchors * num_classes, kernel_size=3, stride=1, padding=1
        )
        modules_list = [self.cls_subnet, self.cls_score]

        # box subnet
        bbox_subnet = []
        cur_channels = in_channels
        for _ in range(num_convs):
            bbox_subnet.append(
                nn.Conv2d(cur_channels, bbox_channels, kernel_size=3, stride=1, padding=1)
            )
            cur_channels = bbox_channels
            bbox_subnet.append(nn.ReLU())

        self.bbox_subnet = nn.Sequential(*bbox_subnet)
        self.bbox_pred = nn.Conv2d(
            cur_channels, num_anchors * 4, kernel_size=3, stride=1, padding=1
        )
        modules_list.extend([self.bbox_subnet, self.bbox_pred])

        # mask subnet
        if self.mask_on:
            mask_subnet = []
            cur_channels = in_channels
            for _ in range(num_convs):
                mask_subnet.append(
                    nn.Conv2d(cur_channels, mask_channels, kernel_size=3, stride=1, padding=1)
                )
                cur_channels = mask_channels
                mask_subnet.append(nn.ReLU())

            self.mask_subnet = nn.Sequential(*mask_subnet)
            modules_list.append(self.mask_subnet)
            for mask_size in self.mask_sizes:
                cur_mask_module = "mask_pred_%02d" % mask_size
                self.add_module(
                    cur_mask_module,
                    nn.Conv2d(
                        cur_channels, mask_size * mask_size, kernel_size=1, stride=1, padding=0
                    ),
                )
                modules_list.append(getattr(self, cur_mask_module))
            if self.align_on:
                if self.bipyramid_on:
                    for lvl in range(num_levels):
                        cur_mask_module = "align2nat_%02d" % lvl
                        lambda_val = 2 ** lvl
                        setattr(self, cur_mask_module, SwapAlign2Nat(lambda_val))
                    # Also the fusing layer, stay at the same channel size
                    mask_fuse = [
                        nn.Conv2d(cur_channels, cur_channels, kernel_size=3, stride=1, padding=1),
                        nn.ReLU(),
                    ]
                    self.mask_fuse = nn.Sequential(*mask_fuse)
                    modules_list.append(self.mask_fuse)
                else:
                    self.align2nat = SwapAlign2Nat(1)

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
            pred_deltas (list[Tensor]): #lvl tensors, each has shape (N, Ax4, Hi, Wi).
                The tensor predicts 4-vector (dx,dy,dw,dh) box
                regression values for every anchor. These values are the
                relative offset between the anchor and the ground truth box.
            pred_masks (list(list[Tensor])): #lvl list of tensors, each is a list of
                A tensors of shape (N, M_{i,a}, Hi, Wi).
                The tensor predicts a dense set of M_ixM_i masks at every location.
        """
        pred_logits = [self.cls_score(self.cls_subnet(x)) for x in features]
        pred_deltas = [self.bbox_pred(self.bbox_subnet(x)) for x in features]

        pred_masks = None
        if self.mask_on:
            mask_feats = [self.mask_subnet(x) for x in features]

            if self.bipyramid_on:
                mask_feat_high_res = mask_feats[0]
                H, W = mask_feat_high_res.shape[-2:]
                mask_feats_up = []
                for lvl, mask_feat in enumerate(mask_feats):
                    lambda_val = 2.0 ** lvl
                    mask_feat_up = mask_feat
                    if lvl > 0:
                        mask_feat_up = F.interpolate(
                            mask_feat, scale_factor=lambda_val, mode="bilinear", align_corners=False
                        )
                    mask_feats_up.append(
                        self.mask_fuse(mask_feat_up[:, :, :H, :W] + mask_feat_high_res)
                    )
                mask_feats = mask_feats_up

            pred_masks = []
            for lvl, mask_feat in enumerate(mask_feats):
                cur_masks = []
                for mask_size in self.mask_sizes:
                    cur_mask_module = getattr(self, "mask_pred_%02d" % mask_size)
                    cur_mask = cur_mask_module(mask_feat)
                    if self.align_on:
                        if self.bipyramid_on:
                            cur_mask_module = getattr(self, "align2nat_%02d" % lvl)
                            cur_mask = cur_mask_module(cur_mask)
                        else:
                            cur_mask = self.align2nat(cur_mask)
                    cur_masks.append(cur_mask)
                pred_masks.append(cur_masks)
        return pred_logits, pred_deltas, pred_masks
