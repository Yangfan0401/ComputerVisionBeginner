from typing import Dict, List, Optional
import torch
from common import DetectorBackboneWithFPN, class_spec_nms, get_fpn_location_coords
from torch import nn
from torch.nn import functional as F
from torch.utils.data._utils.collate import default_collate
from torchvision.ops import sigmoid_focal_loss
from one_stage_detector import (
    fcos_apply_deltas_to_locations,
    fcos_get_deltas_from_locations,
    fcos_match_locations_to_gt,
    fcos_make_centerness_targets,
)
import sys


from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from eecs498.grad import reset_seed
from eecs498.utils import detection_visualizer

# Short hand type notation:
TensorDict = Dict[str, torch.Tensor]


class FCOS(nn.Module):
    """
    FCOS: Fully-Convolutional One-Stage Detector

    This class puts together everything you implemented so far. It contains a
    backbone with FPN, and prediction layers (head). It computes loss during
    training and predicts boxes during inference.
    """

    def __init__(self, num_classes: int, fpn_channels: int, stem_channels: List[int]):
        super().__init__()
        self.num_classes = num_classes

        ######################################################################
        # TODO: Initialize backbone and prediction network using arguments.  #
        ######################################################################
        # Feel free to delete these two lines: (but keep variable names same)
        self.backbone = None
        self.pred_net = {}
        # Replace "pass" statement with your code
        self.backbone = DetectorBackboneWithFPN(
            out_channels=fpn_channels
        )  # Result across backone to fpn results

        stem_cls = []
        stem_box = []
        in_channel = fpn_channels
        # Replace "pass" statement with your code
        for chanel in stem_channels:
            cls_conv = nn.Conv2d(in_channel, chanel, kernel_size=3, stride=1, padding=1)
            nn.init.normal_(cls_conv.weight, mean=0, std=0.01)
            nn.init.zeros_(cls_conv.bias)
            stem_cls.append(cls_conv)
            stem_cls.append(nn.ReLU())

            box_conv = nn.Conv2d(in_channel, chanel, kernel_size=3, stride=1, padding=1)
            nn.init.normal_(box_conv.weight, mean=0, std=0.01)
            nn.init.zeros_(box_conv.bias)
            stem_box.append(box_conv)
            stem_box.append(nn.ReLU())
            in_channel = chanel

        last_pred_class = nn.Conv2d(
            stem_channels[-1], num_classes, kernel_size=3, stride=1, padding=1
        )
        nn.init.normal_(last_pred_class.weight, mean=0, std=0.01)
        nn.init.zeros_(last_pred_class.bias)
        pred_cls = nn.Sequential(*stem_cls, last_pred_class)

        last_pred_centerness = nn.Conv2d(
            stem_channels[-1], 1, kernel_size=3, stride=1, padding=1
        )
        nn.init.normal_(last_pred_centerness.weight, mean=0, std=0.01)
        nn.init.zeros_(last_pred_centerness.bias)
        pred_ctr = nn.Sequential(*stem_box, last_pred_centerness)

        last_pred_box = nn.Conv2d(
            stem_channels[-1], 4, kernel_size=3, stride=1, padding=1
        )
        nn.init.normal_(last_pred_box.weight, mean=0, std=0.01)
        nn.init.zeros_(last_pred_box.bias)
        pred_box = nn.Sequential(*stem_box, last_pred_box)

        self.pred_net["pred_cls"] = pred_cls
        self.pred_net["pred_box"] = pred_box
        self.pred_net["pred_ctr"] = pred_ctr
        ######################################################################
        #                           END OF YOUR CODE                         #
        ######################################################################

        # Averaging factor for training loss; EMA of foreground locations.
        # STUDENTS: See its use in `forward` when you implement losses.
        self._normalizer = 150  # per image

    def forward(
        self,
        images: torch.Tensor,
        gt_boxes: Optional[torch.Tensor] = None,
        test_score_thresh: Optional[float] = None,
        test_nms_thresh: Optional[float] = None,
    ):
        """
        Args:
            images: Batch of images, tensors of shape `(B, C, H, W)`.
            gt_boxes: Batch of training boxes, tensors of shape `(B, N, 5)`.
                `gt_boxes[i, j] = (x1, y1, x2, y2, C)` gives information about
                the `j`th object in `images[i]`. The position of the top-left
                corner of the box is `(x1, y1)` and the position of bottom-right
                corner of the box is `(x2, x2)`. These coordinates are
                real-valued in `[H, W]`. `C` is an integer giving the category
                label for this bounding box. Not provided during inference.
            test_score_thresh: During inference, discard predictions with a
                confidence score less than this value. Ignored during training.
            test_nms_thresh: IoU threshold for NMS during inference. Ignored
                during training.

        Returns:
            Losses during training and predictions during inference.
        """

        ######################################################################
        # TODO: Process the image through backbone, FPN, and prediction head #
        # to obtain model predictions at every FPN location.                 #
        # Get dictionaries of keys {"p3", "p4", "p5"} giving predicted class #
        # logits, deltas, and centerness.                                    #
        ######################################################################
        # Feel free to delete this line: (but keep variable names same)
        pred_cls_logits, pred_boxreg_deltas, pred_ctr_logits = {}, {}, {}
        # Replace "pass" statement with your code
        fpn_feats = self.backbone(images)

        for level_name, val in fpn_feats.items():
            pred_cls_logits[level_name] = (
                self.pred_net["pred_cls"](val).flatten(start_dim=2).permute(0, 2, 1)
            )
            pred_ctr_logits[level_name] = (
                self.pred_net["pred_ctr"](val).flatten(start_dim=2).permute(0, 2, 1)
            )
            pred_boxreg_deltas[level_name] = (
                self.pred_net["pred_box"](val).flatten(start_dim=2).permute(0, 2, 1)
            )

        ######################################################################
        # TODO: Get absolute co-ordinates `(xc, yc)` for every location in
        # FPN levels.
        #
        # HINT: You have already implemented everything, just have to
        # call the functions properly.
        ######################################################################
        # Feel free to delete this line: (but keep variable names same)
        locations_per_fpn_level = None
        # Replace "pass" statement with your code
        shape_per_fpn_level = {}
        for level_name, feat in fpn_feats.items():
            shape_per_fpn_level[level_name] = feat.shape
        locations_per_fpn_level = get_fpn_location_coords(
            shape_per_fpn_level, self.backbone.fpn_strides
        )
        ######################################################################
        #                           END OF YOUR CODE                         #
        ######################################################################

        if not self.training:
            # During inference, just go to this method and skip rest of the
            # forward pass.
            # fmt: off
            return self.inference(
                images, locations_per_fpn_level,
                pred_cls_logits, pred_boxreg_deltas, pred_ctr_logits,
                test_score_thresh=test_score_thresh,
                test_nms_thresh=test_nms_thresh,
            )
            # fmt: on

        ######################################################################
        # TODO: Assign ground-truth boxes to feature locations. We have this
        # implemented in a `fcos_match_locations_to_gt`. This operation is NOT
        # batched so call it separately per GT boxes in batch.
        ######################################################################
        # List of dictionaries with keys {"p3", "p4", "p5"} giving matched
        # boxes for locations per FPN level, per image. Fill this list:
        matched_gt_boxes = []
        # Replace "pass" statement with your code
        Num_Batches = gt_boxes.shape[0]
        strides_per_fpn_level = self.backbone.fpn_strides
        for batch in range(Num_Batches):
            matched_gt_boxes.append(
                fcos_match_locations_to_gt(
                    locations_per_fpn_level, strides_per_fpn_level, gt_boxes[batch]
                )
            )

        # Calculate GT deltas for these matched boxes. Similar structure
        # as `matched_gt_boxes` above. Fill this list:
        matched_gt_deltas = []
        # Replace "pass" statement with your code
        for batch in range(Num_Batches):
            deltas = {}
            for level_name, locations in locations_per_fpn_level.items():
                deltas[level_name] = fcos_get_deltas_from_locations(
                    locations,
                    matched_gt_boxes[batch][level_name],
                    strides_per_fpn_level[level_name],
                )
            matched_gt_deltas.append(deltas)
        ######################################################################
        #                           END OF YOUR CODE                         #
        ######################################################################

        # Collate lists of dictionaries, to dictionaries of batched tensors.
        # These are dictionaries with keys {"p3", "p4", "p5"} and values as
        # tensors of shape (batch_size, locations_per_fpn_level, 5 or 4)
        matched_gt_boxes = default_collate(matched_gt_boxes)
        matched_gt_deltas = default_collate(matched_gt_deltas)

        # Combine predictions and GT from across all FPN levels.
        # shape: (batch_size, num_locations_across_fpn_levels, ...)
        matched_gt_boxes = self._cat_across_fpn_levels(matched_gt_boxes)
        matched_gt_deltas = self._cat_across_fpn_levels(matched_gt_deltas)
        pred_cls_logits = self._cat_across_fpn_levels(pred_cls_logits)
        pred_boxreg_deltas = self._cat_across_fpn_levels(pred_boxreg_deltas)
        pred_ctr_logits = self._cat_across_fpn_levels(pred_ctr_logits)

        # Perform EMA update of normalizer by number of positive locations.
        num_pos_locations = (matched_gt_boxes[:, :, 4] != -1).sum()
        pos_loc_per_image = num_pos_locations.item() / images.shape[0]
        self._normalizer = 0.9 * self._normalizer + 0.1 * pos_loc_per_image

        #######################################################################
        # TODO: Calculate losses per location for classification, box reg and
        # centerness. Remember to set box/centerness losses for "background"
        # positions to zero.
        ######################################################################
        # Feel free to delete this line: (but keep variable names same)
        loss_cls, loss_box, loss_ctr = None, None, None

        # Replace "pass" statement with your code
        gt_boxes_class = matched_gt_boxes[:, :, -1]
        target_label = torch.zeros(pred_cls_logits.shape)
        class_mask = gt_boxes_class >= 0
        target_label = torch.zeros_like(pred_cls_logits)

        target_label[class_mask, gt_boxes_class[class_mask].to(dtype=torch.int64)] = 1

        loss_cls = sigmoid_focal_loss(pred_cls_logits, target_label)
        loss_box = 0.25 * F.l1_loss(
            pred_boxreg_deltas, matched_gt_deltas, reduction="none"
        )
        loss_box[gt_boxes_class < 0] *= 0.0

        # 先拉平，但保留原始 deltas
        deltas_flat = matched_gt_deltas.view(-1, 4)
        ctr_targets = fcos_make_centerness_targets(deltas_flat)  # 背景位置会得到 -1
        # 获取正样本掩码（与分类、回归一致）
        pos_mask = matched_gt_boxes[:, :, 4].view(-1) >= 0
        # 将背景目标值改为 0（避免 BCE 报错）
        ctr_targets[~pos_mask] = 0.0
        # 计算损失
        loss_ctr = F.binary_cross_entropy_with_logits(
            pred_ctr_logits.view(-1), ctr_targets, reduction="none"
        )
        # 用掩码剔除背景
        loss_ctr[~pos_mask] *= 0.0
        ######################################################################
        #                            END OF YOUR CODE                        #
        ######################################################################
        # Sum all locations and average by the EMA of foreground locations.
        # In training code, we simply add these three and call `.backward()`
        return {
            "loss_cls": loss_cls.sum() / (self._normalizer * images.shape[0]),
            "loss_box": loss_box.sum() / (self._normalizer * images.shape[0]),
            "loss_ctr": loss_ctr.sum() / (self._normalizer * images.shape[0]),
        }

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

    def inference(
        self,
        images: torch.Tensor,
        locations_per_fpn_level: Dict[str, torch.Tensor],
        pred_cls_logits: Dict[str, torch.Tensor],
        pred_boxreg_deltas: Dict[str, torch.Tensor],
        pred_ctr_logits: Dict[str, torch.Tensor],
        test_score_thresh: float = 0.3,
        test_nms_thresh: float = 0.5,
    ):
        """
        Run inference on a single input image (batch size = 1). Other input
        arguments are same as those computed in `forward` method. This method
        should not be called from anywhere except from inside `forward`.

        Returns:
            Three tensors:
                - pred_boxes: Tensor of shape `(N, 4)` giving *absolute* XYXY
                  co-ordinates of predicted boxes.

                - pred_classes: Tensor of shape `(N, )` giving predicted class
                  labels for these boxes (one of `num_classes` labels). Make
                  sure there are no background predictions (-1).

                - pred_scores: Tensor of shape `(N, )` giving confidence scores
                  for predictions: these values are `sqrt(class_prob * ctrness)`
                  where class_prob and ctrness are obtained by applying sigmoid
                  to corresponding logits.
        """

        # Gather scores and boxes from all FPN levels in this list. Once
        # gathered, we will perform NMS to filter highly overlapping predictions.
        pred_boxes_all_levels = []
        pred_classes_all_levels = []
        pred_scores_all_levels = []

        for level_name in locations_per_fpn_level.keys():
            # Get locations and predictions from a single level.
            # We index predictions by `[0]` to remove batch dimension.
            level_locations = locations_per_fpn_level[level_name]
            level_cls_logits = pred_cls_logits[level_name][0]
            level_deltas = pred_boxreg_deltas[level_name][0]
            level_ctr_logits = pred_ctr_logits[level_name][0]

            ##################################################################
            # TODO: FCOS uses the geometric mean of class probability and
            # centerness as the final confidence score. This helps in getting
            # rid of excessive amount of boxes far away from object centers.
            # Compute this value here (recall sigmoid(logits) = probabilities)
            #
            # Then perform the following steps in order:
            #   1. Get the most confidently predicted class and its score for
            #      every box. Use level_pred_scores: (N, num_classes) => (N, )
            #   2. Only retain prediction that have a confidence score higher
            #      than provided threshold in arguments.
            #   3. Obtain predicted boxes using predicted deltas and locations
            #   4. Clip XYXY box-cordinates that go beyond thr height and
            #      and width of input image.
            ##################################################################
            # Feel free to delete this line: (but keep variable names same)
            level_pred_boxes, level_pred_classes, level_pred_scores = (
                None,
                None,
                None,  # Need tensors of shape: (N, 4) (N, ) (N, )
            )

            # Compute geometric mean of class logits and centerness:
            level_pred_scores = torch.sqrt(
                level_cls_logits.sigmoid_() * level_ctr_logits.sigmoid_()
            )
            # Step 1:
            # Replace "pass" statement with your code
            level_pred_classes = level_pred_scores.argmax(dim=1)
            level_pred_scores = level_pred_scores.max(dim=1).values
            # Step 2:
            # Replace "pass" statement with your code
            mask = level_pred_scores > test_score_thresh
            level_pred_scores = level_pred_scores[mask]
            level_pred_classes = level_pred_classes[mask]
            level_locations = level_locations[mask]
            level_deltas = level_deltas[mask]
            # Step 3:
            # Replace "pass" statement with your code
            stride = self.backbone.fpn_strides[level_name]
            level_pred_boxes = fcos_apply_deltas_to_locations(
                level_deltas, level_locations, stride
            )
            # Step 4: Use `images` to get (height, width) for clipping.
            # Replace "pass" statement with your code

            N = level_pred_boxes.shape[0]
            if N != 0:
                _, _, H, W = images.shape
                level_pred_boxes[:, [0, 2]] = level_pred_boxes[:, [0, 2]].clamp(
                    min=0, max=W
                )
                level_pred_boxes[:, [1, 3]] = level_pred_boxes[:, [1, 3]].clamp(
                    min=0, max=H
                )

            ##################################################################
            #                          END OF YOUR CODE                      #
            ##################################################################

            pred_boxes_all_levels.append(level_pred_boxes)
            pred_classes_all_levels.append(level_pred_classes)
            pred_scores_all_levels.append(level_pred_scores)

        ######################################################################
        # Combine predictions from all levels and perform NMS.
        pred_boxes_all_levels = torch.cat(pred_boxes_all_levels)
        pred_classes_all_levels = torch.cat(pred_classes_all_levels)
        pred_scores_all_levels = torch.cat(pred_scores_all_levels)

        # STUDENTS: This function depends on your implementation of NMS.
        keep = class_spec_nms(
            pred_boxes_all_levels,
            pred_scores_all_levels,
            pred_classes_all_levels,
            iou_threshold=test_nms_thresh,
        )
        pred_boxes_all_levels = pred_boxes_all_levels[keep]
        pred_classes_all_levels = pred_classes_all_levels[keep]
        pred_scores_all_levels = pred_scores_all_levels[keep]
        return (
            pred_boxes_all_levels,
            pred_classes_all_levels,
            pred_scores_all_levels,
        )

