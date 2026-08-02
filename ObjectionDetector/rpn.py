import torchvision
from common import get_fpn_location_coords
from two_stage_detector import sample_rpn_training, mix_gt_with_proposals
from two_stage_detector import (
    RPNPredictionNetwork,
    generate_fpn_anchors,
    iou,
    rcnn_apply_deltas_to_anchors,
    rcnn_get_deltas_from_anchors,
    rcnn_match_anchors_to_gt,
)
from typing import Dict, List, Optional, Tuple
from torch.nn import functional as F
import torch
import torch.nn as nn
from a4_helper import train_detector
from common import DetectorBackboneWithFPN
import random
from one_stage_detector import fcos_get_deltas_from_locations
from a4_helper import VOC2007DetectionTiny
from torchvision import transforms
import sys
from pathlib import Path
import math

sys.path.append(str(Path(__file__).parent.parent))
from eecs498.grad import reset_seed, rel_error
from eecs498.utils import detection_visualizer

TensorDict = Dict[str, torch.Tensor]


class RPN(nn.Module):
    """
    Region Proposal Network: First stage of Faster R-CNN detector.

    This class puts together everything you implemented so far. It accepts FPN
    features as input and uses `RPNPredictionNetwork` to predict objectness and
    box reg deltas. Computes proposal boxes for second stage (during both
    training and inference) and losses during training.
    """

    def __init__(
        self,
        fpn_channels: int,
        stem_channels: List[int],
        batch_size_per_image: int,
        anchor_stride_scale: int = 8,
        anchor_aspect_ratios: List[int] = [0.5, 1.0, 2.0],
        anchor_iou_thresholds: Tuple[int, int] = (0.3, 0.6),
        nms_thresh: float = 0.7,
        pre_nms_topk: int = 400,
        post_nms_topk: int = 100,
    ):
        """
        Args:
            batch_size_per_image: Anchors per image to sample for training.
            nms_thresh: IoU threshold for NMS - unlike FCOS, this is used
                during both, training and inference.
            pre_nms_topk: Number of top-K proposals to select before applying
                NMS, per FPN level. This helps in speeding up NMS.
            post_nms_topk: Number of top-K proposals to select after applying
                NMS, per FPN level. NMS is obviously going to be class-agnostic.

        Refer explanations of remaining args in the classes/functions above.
        """
        super().__init__()
        self.pred_net = RPNPredictionNetwork(
            fpn_channels, stem_channels, num_anchors=len(anchor_aspect_ratios)
        )
        # Record all input arguments:
        self.batch_size_per_image = batch_size_per_image
        self.anchor_stride_scale = anchor_stride_scale
        self.anchor_aspect_ratios = anchor_aspect_ratios
        self.anchor_iou_thresholds = anchor_iou_thresholds
        self.nms_thresh = nms_thresh
        self.pre_nms_topk = pre_nms_topk
        self.post_nms_topk = post_nms_topk

    def forward(
        self,
        feats_per_fpn_level: TensorDict,
        strides_per_fpn_level: TensorDict,
        gt_boxes: Optional[torch.Tensor] = None,
    ):
        # Get batch size from FPN feats:
        num_images = feats_per_fpn_level["p3"].shape[0]

        ######################################################################
        # TODO: Implement the training forward pass. Follow these steps:
        #   1. Pass the FPN features per level to the RPN prediction network.
        #   2. Generate anchor boxes for all FPN levels.
        #
        # HINT: You have already implemented everything, just have to call the
        # appropriate functions.
        ######################################################################
        # Feel free to delete this line: (but keep variable names same)
        pred_obj_logits = {}
        pred_boxreg_deltas = {}
        anchors_per_fpn_level = {}
        # Replace "pass" statement with your code
        shape_per_fpn_level = {}
        pred_obj_logits, pred_boxreg_deltas = self.pred_net(
            feats_per_fpn_level
        )
        for level_name, fpn_feats in feats_per_fpn_level.items():
            shape_per_fpn_level[level_name] = fpn_feats.shape

        locations_per_fpn_level = get_fpn_location_coords(
            shape_per_fpn_level, strides_per_fpn_level
        )
        anchors_per_fpn_level = generate_fpn_anchors(
            locations_per_fpn_level,
            strides_per_fpn_level,
            self.anchor_stride_scale,
            self.anchor_aspect_ratios,
        )
        ######################################################################
        #                           END OF YOUR CODE                         #
        ######################################################################

        # We will fill three values in this output dict - "proposals",
        # "loss_rpn_box" (training only), "loss_rpn_obj" (training only)
        output_dict = {}

        # Get image height and width according to feature sizes and strides.
        # We need these to clamp proposals (These should be (224, 224) but we
        # avoid hard-coding them).
        img_h = feats_per_fpn_level["p3"].shape[2] * strides_per_fpn_level["p3"]
        img_w = feats_per_fpn_level["p3"].shape[3] * strides_per_fpn_level["p3"]

        # STUDENT: Implement this method before moving forward with the rest
        # of this `forward` method.
        output_dict["proposals"] = self.predict_proposals(
            anchors_per_fpn_level,
            pred_obj_logits,
            pred_boxreg_deltas,
            (img_w, img_h),
        )
        # Return here during inference - loss computation not required.
        if not self.training:
            return output_dict

        # ... otherwise continue loss computation:
        ######################################################################
        # Match the generated anchors with provided GT boxes. This
        # function is not batched so you may use a for-loop, like FCOS.
        ######################################################################
        # Combine anchor boxes from all FPN levels - we do not need any
        # distinction of boxes across different levels (for training).
        anchor_boxes = self._cat_across_fpn_levels(anchors_per_fpn_level, dim=0)

        # Get matched GT boxes (list of B tensors, each of shape `(H*W*A, 5)`
        # giving matching GT boxes to anchor boxes). Fill this list:
        matched_gt_boxes = []
        # Replace "pass" statement with your code
        for batch in range(num_images):
            matched_gt_boxes.append(
                 rcnn_match_anchors_to_gt(
                    anchor_boxes, gt_boxes[batch], self.anchor_iou_thresholds
                )
            )
        ######################################################################
        #                           END OF YOUR CODE                         #
        ######################################################################

        # Combine matched boxes from all images to a `(B, HWA, 5)` tensor.
        matched_gt_boxes = torch.stack(matched_gt_boxes, dim=0)

        # Combine predictions across all FPN levels.
        pred_obj_logits = self._cat_across_fpn_levels(pred_obj_logits)
        pred_boxreg_deltas = self._cat_across_fpn_levels(pred_boxreg_deltas)

        if self.training:
            # Repeat anchor boxes `batch_size` times so there is a 1:1
            # correspondence with GT boxes.
            anchor_boxes = anchor_boxes.unsqueeze(0).repeat(num_images, 1, 1)
            anchor_boxes = anchor_boxes.contiguous().view(-1, 4)

            # Collapse `batch_size`, and `HWA` to a single dimension so we have
            # simple `(-1, 4 or 5)` tensors. This simplifies loss computation.
            matched_gt_boxes = matched_gt_boxes.view(-1, 5)
            pred_obj_logits = pred_obj_logits.view(-1)
            pred_boxreg_deltas = pred_boxreg_deltas.view(-1, 4)

            ##################################################################
            # TODO: Compute training losses. Follow three steps in order:
            #   1. Sample a few anchor boxes for training. Pass the variable
            #      `matched_gt_boxes` to `sample_rpn_training` function and
            #      use those indices to get subset of predictions and targets.
            #      RPN samples 50-50% foreground/background anchors, unless
            #      there aren't enough foreground anchors.
            #
            #   2. Compute GT targets for box regression (you have implemented
            #      the transformation function already).
            #
            #   3. Calculate objectness and box reg losses per sampled anchor.
            #      Remember to set box loss for "background" anchors to 0.
            ##################################################################
            # Feel free to delete this line: (but keep variable names same)
            loss_obj, loss_box = None, None
            # Replace "pass" statement with your code
            num_samples = self.batch_size_per_image
            fg_idx, bg_idx = sample_rpn_training(matched_gt_boxes, num_samples, 0.5)
            sample_idx = torch.concat([fg_idx, bg_idx])
            anchor_boxes = anchor_boxes[sample_idx]
            matched_gt_boxes = matched_gt_boxes[sample_idx]
            pred_obj_logits = pred_obj_logits[sample_idx]
            pred_boxreg_deltas = pred_boxreg_deltas[sample_idx]

            gt_deltas = rcnn_get_deltas_from_anchors(anchor_boxes, matched_gt_boxes)
            loss_box = F.l1_loss(pred_boxreg_deltas, gt_deltas, reduction="none")
            loss_box[gt_deltas == -1e8] *= 0.0
            
            gt_objectness = (matched_gt_boxes[:, :4] != -1).all(dim=1).to(dtype=torch.float)
            loss_obj = F.binary_cross_entropy_with_logits(
                pred_obj_logits, gt_objectness, reduction="none"
            )
            ##################################################################
            #                         END OF YOUR CODE                       #
            ##################################################################

            # Sum losses and average by num(foreground + background) anchors.
            # In training code, we simply add these two and call `.backward()`
            total_batch_size = self.batch_size_per_image * num_images
            output_dict["loss_rpn_obj"] = loss_obj.sum() / total_batch_size
            output_dict["loss_rpn_box"] = loss_box.sum() / total_batch_size

        return output_dict

    @torch.no_grad()  # Don't track gradients in this function.
    def predict_proposals(
        self,
        anchors_per_fpn_level: Dict[str, torch.Tensor],
        pred_obj_logits: Dict[str, torch.Tensor],
        pred_boxreg_deltas: Dict[str, torch.Tensor],
        image_size: Tuple[int, int],  # (width, height)
    ):
        """
        Predict proposals for a batch of images for the second stage. Other
        input arguments are same as those computed in `forward` method. This
        method should not be called from anywhere except from inside `forward`.

        Returns:
            torch.Tensor
                proposals: Tensor of shape `(keep_topk, 4)` giving *absolute*
                    XYXY co-ordinates of predicted proposals. These will serve
                    as anchor boxes for the second stage.
        """

        # Gather proposals from all FPN levels in this list.
        proposals_all_levels = {
            level_name: None for level_name, _ in anchors_per_fpn_level.items()
        }
        for level_name in anchors_per_fpn_level.keys():
            # Get anchor boxes and predictions from a single level.
            level_anchors = anchors_per_fpn_level[level_name]

            # shape: (batch_size, HWA), (batch_size, HWA, 4)
            level_obj_logits = pred_obj_logits[level_name]
            level_boxreg_deltas = pred_boxreg_deltas[level_name]

            # Fill proposals per image, for this FPN level, in this list.
            level_proposals_per_image = []
            for _batch_idx in range(level_obj_logits.shape[0]):
                ##############################################################
                # TODO: Perform the following steps in order:
                #   1. Transform the anchors to proposal boxes using predicted
                #      box deltas, clamp to image height and width.
                #   2. Sort all proposals by their predicted objectness, and
                #      retain `self.pre_nms_topk` proposals. This speeds up
                #      our NMS computation. HINT: `torch.topk`
                #   3. Apply NMS and retain `keep_topk_per_level` proposals
                #      per image, per level.
                #
                # NOTE: Your `nms` method may be slow for training - you may
                # use `torchvision.ops.nms` with exact same input arguments,
                # to speed up training. We will grade your `nms` implementation
                # separately; you will NOT lose points if you don't use it here.
                #
                # Note that deltas, anchor boxes, and objectness logits have
                # different shapes, you need to make some intermediate views.
                ##############################################################
                # Replace "pass" statement with your code
                w, h = image_size
                proposal = rcnn_apply_deltas_to_anchors(
                    level_boxreg_deltas[_batch_idx], level_anchors
                )
                proposal[:, [0, 2]] = proposal[:, [0, 2]].clamp(min=0, max=w)
                proposal[:, [1, 3]] = proposal[:, [1, 3]].clamp(min=0, max=h)
                
                num_boxes = level_obj_logits[_batch_idx].shape[0]
                k = min(self.pre_nms_topk, num_boxes)
                indices = torch.topk(level_obj_logits[_batch_idx], k).indices
                proposal = proposal[indices]
                scores = level_obj_logits[_batch_idx][indices]
                
                keep = torchvision.ops.nms(proposal, scores, self.nms_thresh)
                proposal = proposal[keep]
                scores = scores[keep]
                num_boxes = scores.shape[0]
                
                k = min(self.post_nms_topk, num_boxes)
                keep_topk_per_level = torch.topk(scores, k).indices
                proposal = proposal[keep_topk_per_level]
                level_proposals_per_image.append(proposal)
                ##############################################################
                #                        END OF YOUR CODE                    #
                ##############################################################

            # Collate proposals from individual images. Do not stack these
            # tensors, they may have different shapes since few images or
            # levels may have less than `post_nms_topk` proposals. We could
            # pad these tensors but there's no point - they will be used by
            # `torchvision.ops.roi_align` in second stage which operates
            # with lists, not batched tensors.
            proposals_all_levels[level_name] = level_proposals_per_image

        return proposals_all_levels

    @staticmethod
    def _cat_across_fpn_levels(
            dict_with_fpn_levels: Dict[str, torch.Tensor], dim: int = 1
    ):
        """
        Convert a dict of tensors across FPN levels {"p3", "p4", "p5"} to a
        single tensor. Values could be anything - batches of image features,
        GT targets, etc.
        """
        return torch.cat(list(dict_with_fpn_levels.values()), dim=dim)
