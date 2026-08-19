from typing import Dict, List
import torch
from torchvision import transforms
import torch.nn as nn
import os
from PIL import Image
import json
TensorDict = Dict[str, torch.Tensor]

@torch.no_grad()
def fcos_match_locations_to_gt(
    locations_per_fpn_level: TensorDict,
    strides_per_fpn_level: Dict[str, int],
    gt_boxes: torch.Tensor,
) -> TensorDict:
    """
    Match centers of the locations of FPN feature with a set of GT bounding
    boxes of the input image. Since our model makes predictions at every FPN
    feature map location, we must supervise it with an appropriate GT box.
    There are multiple GT boxes in image, so FCOS has a set of heuristics to
    assign centers with GT, which we implement here.

    NOTE: This function is NOT BATCHED. Call separately for GT box batches.

    Args:
        locations_per_fpn_level: Centers at different levels of FPN (p3, p4, p5),
            that are already projected to absolute co-ordinates in input image
            dimension. Dictionary of three keys: (p3, p4, p5) giving tensors of
            shape `(H * W, 2)` where H = W is the size of feature map.
        strides_per_fpn_level: Dictionary of same keys as above, each with an
            integer value giving the stride of corresponding FPN level.
            See `common.py` for more details.
        gt_boxes: GT boxes of a single image, a batch of `(M, 5)` boxes with
            absolute co-ordinates and class ID `(x1, y1, x2, y2, C)`. In this
            codebase, this tensor is directly served by the dataloader.

    Returns:
        Dict[str, torch.Tensor]
            Dictionary with same keys as `shape_per_fpn_level` and values as
            tensors of shape `(N, 5)` GT boxes, one for each center. They are
            one of M input boxes, or a dummy box called "background" that is
            `(-1, -1, -1, -1, -1)`. Background indicates that the center does
            not belong to any object.
    """

    matched_gt_boxes = {
        level_name: None for level_name in locations_per_fpn_level.keys()
    }

    # Do this matching individually per FPN level.
    for level_name, centers in locations_per_fpn_level.items():

        # Get stride for this FPN level.
        stride = strides_per_fpn_level[level_name]

        x, y = centers.unsqueeze(dim=2).unbind(dim=1)
        x0, y0, x1, y1 = gt_boxes[:, :4].unsqueeze(dim=0).unbind(dim=2)
        pairwise_dist = torch.stack([x - x0, y - y0, x1 - x, y1 - y], dim=2)

        # Pairwise distance between every feature center and GT box edges:
        # shape: (num_gt_boxes, num_centers_this_level, 4)
        pairwise_dist = pairwise_dist.permute(1, 0, 2)

        # The original FCOS anchor matching rule: anchor point must be inside GT.
        match_matrix = pairwise_dist.min(dim=2).values > 0

        # Multilevel anchor matching in FCOS: each anchor is only responsible
        # for certain scale range.
        # Decide upper and lower bounds of limiting targets.
        pairwise_dist = pairwise_dist.max(dim=2).values

        lower_bound = stride * 4 if level_name != "p3" else 0
        upper_bound = stride * 8 if level_name != "p5" else float("inf")
        match_matrix &= (pairwise_dist > lower_bound) & (
            pairwise_dist < upper_bound
        )

        # Match the GT box with minimum area, if there are multiple GT matches.
        gt_areas = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (
            gt_boxes[:, 3] - gt_boxes[:, 1]
        )

        # Get matches and their labels using match quality matrix.
        match_matrix = match_matrix.to(torch.float32)
        match_matrix *= 1e8 - gt_areas[:, None]

        # Find matched ground-truth instance per anchor (un-matched = -1).
        match_quality, matched_idxs = match_matrix.max(dim=0)
        matched_idxs[match_quality < 1e-5] = -1

        # Anchors with label 0 are treated as background.
        matched_boxes_this_level = gt_boxes[matched_idxs.clip(min=0)]
        matched_boxes_this_level[matched_idxs < 0, :] = -1

        matched_gt_boxes[level_name] = matched_boxes_this_level

    return matched_gt_boxes


def fcos_get_deltas_from_locations(
    locations: torch.Tensor, gt_boxes: torch.Tensor, stride: int
) -> torch.Tensor:
    """
    Compute distances from feature locations to GT box edges. These distances
    are called "deltas" - `(left, top, right, bottom)` or simply `LTRB`. The
    feature locations and GT boxes are given in absolute image co-ordinates.

    These deltas are used as targets for training FCOS to perform box regression
    and centerness regression. They must be "normalized" by the stride of FPN
    feature map (from which feature locations were computed, see the function
    `get_fpn_location_coords`). If GT boxes are "background", then deltas must
    be `(-1, -1, -1, -1)`.

    NOTE: This transformation function should not require GT class label. Your
    implementation must work for GT boxes being `(N, 4)` or `(N, 5)` tensors -
    without or with class labels respectively. You may assume that all the
    background boxes will be `(-1, -1, -1, -1)` or `(-1, -1, -1, -1, -1)`.

    Args:
        locations: Tensor of shape `(N, 2)` giving `(xc, yc)` feature locations.
        gt_boxes: Tensor of shape `(N, 4 or 5)` giving GT boxes.
        stride: Stride of the FPN feature map.

    Returns:
        torch.Tensor
            Tensor of shape `(N, 4)` giving deltas from feature locations, that
            are normalized by feature stride.
    """
    ##########################################################################
    # TODO: Implement the logic to get deltas from feature locations.        #
    ##########################################################################
    # Set this to Tensor of shape (N, 4) giving deltas (left, top, right, bottom)
    # from the locations to GT box edges, normalized by FPN stride.
    deltas = None

    # Replace "pass" statement with your code
    x, y = locations.unbind(dim=1)
    x1, y1, x2, y2 = gt_boxes[:, :4].unbind(dim=1)
    deltas = torch.stack([
        x - x1,
        y - y1,
        x2 - x,
        y2 - y
    ], dim=1) / stride

    bg_mask = (gt_boxes[:,:4] == -1).all(dim=1)
    deltas[bg_mask] = -1
    ##########################################################################
    #                             END OF YOUR CODE                           #
    ##########################################################################

    return deltas


def fcos_apply_deltas_to_locations(
    deltas: torch.Tensor, locations: torch.Tensor, stride: int
) -> torch.Tensor:
    """
    Implement the inverse of `fcos_get_deltas_from_locations` here:

    Given edge deltas (left, top, right, bottom) and feature locations of FPN, get
    the resulting bounding box co-ordinates by applying deltas on locations. This
    method is used for inference in FCOS: deltas are outputs from model, and
    applying them to anchors will give us final box predictions.

    Recall in above method, we were required to normalize the deltas by feature
    stride. Similarly, we have to un-normalize the input deltas with feature
    stride before applying them to locations, because the given input locations are
    already absolute co-ordinates in image dimensions.

    Args:
        deltas: Tensor of shape `(N, 4)` giving edge deltas to apply to locations.
        locations: Locations to apply deltas on. shape: `(N, 2)`
        stride: Stride of the FPN feature map.

    Returns:
        torch.Tensor
            Same shape as deltas and locations, giving co-ordinates of the
            resulting boxes `(x1, y1, x2, y2)`, absolute in image dimensions.
    """
    ##########################################################################
    # TODO: Implement the transformation logic to get boxes.                 #
    #                                                                        #
    # NOTE: The model predicted deltas MAY BE negative, which is not valid   #
    # for our use-case because the feature center must lie INSIDE the final  #
    # box. Make sure to clip them to zero.                                   #
    ##########################################################################
    # Replace "pass" statement with your code
    deltas = deltas.clamp(min=0)
    deltas *= stride
    xc, yc = locations.unbind(dim=1)
    l, t, r, b = deltas.unbind(dim=1)
    x1 = xc - l
    y1 = yc - t

    x2 = xc + r
    y2 = yc + b
    output_boxes = torch.stack([x1, y1, x2, y2], dim=1)
    ##########################################################################
    #                             END OF YOUR CODE                           #
    ##########################################################################

    return output_boxes


def fcos_make_centerness_targets(deltas: torch.Tensor):
    """
    Given LTRB deltas of GT boxes, compute GT targets for supervising the
    centerness regression predictor. See `fcos_get_deltas_from_locations` on
    how deltas are computed. If GT boxes are "background" => deltas are
    `(-1, -1, -1, -1)`, then centerness should be `-1`.

    For reference, centerness equation is available in FCOS paper
    https://arxiv.org/abs/1904.01355 (Equation 3).

    Args:
        deltas: Tensor of shape `(N, 4)` giving LTRB deltas for GT boxes.

    Returns:
        torch.Tensor
            Tensor of shape `(N, )` giving centerness regression targets.
    """
    ##########################################################################
    # TODO: Implement the centerness calculation logic.                      #
    ##########################################################################
    centerness = None
    # Replace "pass" statement with your code
    l, t, r, b = deltas[:].unbind(dim=1)
    out = torch.stack([l, r, t, b], dim=1)
    min_lr = out[torch.arange(out.shape[0]), out[:,:2].argmin(dim=1)]
    min_tb = out[torch.arange(out.shape[0]), 2 + out[:,-2:].argmin(dim=1)]
    max_lr = out[torch.arange(out.shape[0]), out[:,:2].argmax(dim=1)]
    max_tb = out[torch.arange(out.shape[0]), 2 + out[:,-2:].argmax(dim=1)]
    centerness = (min_lr * min_tb) / (max_lr * max_tb)
    centerness = centerness.sqrt_()
    bg_mask = (deltas[:, :] == -1).all(dim=1)
    centerness[bg_mask] = -1
    ##########################################################################
    #                             END OF YOUR CODE                           #
    ##########################################################################

    return centerness


class VOC2007DetectionTiny(torch.utils.data.Dataset):
    """
    A tiny version of PASCAL VOC 2007 Detection dataset that includes images and
    annotations with small images and no difficult boxes.
    """

    def __init__(
        self,
        dataset_dir: str,
        split: str = "train",
        download: bool = False,
        image_size: int = 224,
    ):
        """
        Args:
            download: Whether to download PASCAL VOC 2007 to `dataset_dir`.
            image_size: Size of imges in the batch. The shorter edge of images
                will be resized to this size, followed by a center crop. For
                val, center crop will not be taken to capture all detections.
        """
        super().__init__()
        self.image_size = image_size

        # Attempt to download the dataset from Justin's server:
        if download:
            self._attempt_download(dataset_dir)

        # fmt: off
        voc_classes = [
            "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
            "car", "cat", "chair", "cow", "diningtable", "dog",
            "horse", "motorbike", "person", "pottedplant", "sheep",
            "sofa", "train", "tvmonitor"
        ]
        # fmt: on

        # Make a (class to ID) and inverse (ID to class) mapping.
        self.class_to_idx = {
            _class: _idx for _idx, _class in enumerate(voc_classes)
        }
        self.idx_to_class = {
            _idx: _class for _idx, _class in enumerate(voc_classes)
        }

        # Load instances from JSON file:
        self.instances = json.load(
            open(os.path.join(dataset_dir, f"voc07_{split}.json"))
        )
        self.dataset_dir = dataset_dir

        # Define a transformation function for image: Resize the shorter image
        # edge then take a center crop (optional) and normalize.
        _transforms = [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
        self.image_transform = transforms.Compose(_transforms)

    @staticmethod
    def _attempt_download(dataset_dir: str):
        """
        Try to download VOC dataset and save it to `dataset_dir`.
        """
        import wget

        os.makedirs(dataset_dir, exist_ok=True)
        
        import ssl
        ssl._create_default_https_context = ssl._create_unverified_context

# 然后正常下载
        # fmt: off
        wget.download(
            "https://web.eecs.umich.edu/~justincj/data/VOCtrainval_06-Nov-2007.tar",
            out=dataset_dir,
        )
        wget.download(
            "https://web.eecs.umich.edu/~justincj/data/voc07_train.json",
            out=dataset_dir,
        )
        wget.download(
            "https://web.eecs.umich.edu/~justincj/data/voc07_val.json",
            out=dataset_dir,
        )
        # fmt: on

        # Extract TAR file:
        import tarfile

        voc_tar = tarfile.open(
            os.path.join(dataset_dir, "VOCtrainval_06-Nov-2007.tar")
        )
        voc_tar.extractall(dataset_dir)
        voc_tar.close()

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, index: int):
        # PIL image and dictionary of annotations.
        image_path, ann = self.instances[index]
        # TODO: Remove this after the JSON files are fixed on Justin's server:
        image_path = image_path.replace("./here/", "")
        image_path = os.path.join(self.dataset_dir, image_path)
        image = Image.open(image_path).convert("RGB")

        # Collect a list of GT boxes: (N, 4), and GT classes: (N, )
        gt_boxes = torch.tensor([inst["xyxy"] for inst in ann])
        gt_classes = torch.Tensor([self.class_to_idx[inst["name"]] for inst in ann])
        gt_classes = gt_classes.unsqueeze(1)  # (N, 1)

        # Record original image size before transforming.
        original_width, original_height = image.size

        # Normalize bounding box co-ordinates to bring them in [0, 1]. This is
        # temporary, simply to ease the transformation logic.
        normalize_tens = torch.tensor(
            [original_width, original_height, original_width, original_height]
        )
        gt_boxes /= normalize_tens[None, :]

        # Transform input image to CHW tensor.
        image = self.image_transform(image)

        # WARN: Even dimensions should be even numbers else it messes up
        # upsampling in FPN.

        # Apply image resizing transformation to bounding boxes.
        if self.image_size is not None:
            if original_height >= original_width:
                new_width = self.image_size
                new_height = original_height * self.image_size / original_width
            else:
                new_height = self.image_size
                new_width = original_width * self.image_size / original_height

            _x1 = (new_width - self.image_size) // 2
            _y1 = (new_height - self.image_size) // 2

            # Un-normalize bounding box co-ordinates and shift due to center crop.
            # Clamp to (0, image size).
            gt_boxes[:, 0] = torch.clamp(gt_boxes[:, 0] * new_width - _x1, min=0)
            gt_boxes[:, 1] = torch.clamp(gt_boxes[:, 1] * new_height - _y1, min=0)
            gt_boxes[:, 2] = torch.clamp(
                gt_boxes[:, 2] * new_width - _x1, max=self.image_size
            )
            gt_boxes[:, 3] = torch.clamp(
                gt_boxes[:, 3] * new_height - _y1, max=self.image_size
            )

        # Concatenate GT classes with GT boxes; shape: (N, 5)
        gt_boxes = torch.cat([gt_boxes, gt_classes], dim=1)

        # Center cropping may completely exclude certain boxes that were close
        # to image boundaries. Set them to -1
        invalid = (gt_boxes[:, 0] > gt_boxes[:, 2]) | (
            gt_boxes[:, 1] > gt_boxes[:, 3]
        )
        gt_boxes[invalid] = -1

        # Pad to max 40 boxes, that's enough for VOC.
        gt_boxes = torch.cat(
            [gt_boxes, torch.zeros(40 - len(gt_boxes), 5).fill_(-1.0)]
        )
        # Return image path because it is needed for evaluation.
        return image_path, image, gt_boxes


class FCOSPredictionNetwork(nn.Module):
    """
    FCOS prediction network that accepts FPN feature maps from different levels
    and makes three predictions at every location: bounding boxes, class ID and
    centerness. This module contains a "stem" of convolution layers, along with
    one final layer per prediction. For a visual depiction, see Figure 2 (right
    side) in FCOS paper: https://arxiv.org/abs/1904.01355

    We will use feature maps from FPN levels (P3, P4, P5) and exclude (P6, P7).
    """

    def __init__(
        self, num_classes: int, in_channels: int, stem_channels: List[int]
    ):
        """
        Args:
            num_classes: Number of object classes for classification.
            in_channels: Number of channels in input feature maps. This value
                is same as the output channels of FPN, since the head directly
                operates on them.
            stem_channels: List of integers giving the number of output channels
                in each convolution layer of stem layers.
        """
        super().__init__()

        ######################################################################
        # TODO: Create a stem of alternating 3x3 convolution layers and RELU
        # activation modules. Note there are two separate stems for class and
        # box stem. The prediction layers for box regression and centerness
        # operate on the output of `stem_box`.
        # See FCOS figure again; both stems are identical.
        # 
        # Use `in_channels` and `stem_channels` for creating these layers, the
        # docstring above tells you what they mean. Initialize weights of each
        # conv layer from a normal distribution with mean = 0 and std dev = 0.01
        # and all biases with zero. Use conv stride = 1 and zero padding such
        # that size of input features remains same: remember we need predictions
        # at every location in feature map, we shouldn't "lose" any locations.
        ######################################################################
        # Fill these.
        stem_cls = []
        stem_box = []
        # Replace "pass" statement with your code
        for stem_channel in stem_channels:
            cls_conv = nn.Conv2d(in_channels, stem_channel, kernel_size=3, stride=1, padding=1)
            nn.init.xavier_normal(cls_conv.weight)
            nn.init.zeros_(cls_conv.bias)
            stem_cls.append(cls_conv)
            stem_cls.append(nn.ReLU())
            
            box_conv = nn.Conv2d(in_channels, stem_channel, kernel_size=3, stride=1, padding=1)
            nn.init.xavier_normal(box_conv.weight)
            nn.init.zeros_(box_conv.bias)
            stem_box.append(box_conv)
            stem_box.append(nn.ReLU())
            in_channels = stem_channel

        # Wrap the layers defined by student into a `nn.Sequential` module:
        self.stem_cls = nn.Sequential(*stem_cls)
        self.stem_box = nn.Sequential(*stem_box)

        ######################################################################
        # TODO: Create THREE 3x3 conv layers for individually predicting three
        # things at every location of feature map:
        #     1. object class logits (`num_classes` outputs)
        #     2. box regression deltas (4 outputs: LTRB deltas from locations)
        #     3. centerness logits (1 output)
        #
        # Class probability and actual centerness are obtained by applying
        # sigmoid activation to these logits. However, DO NOT initialize those
        # modules here. This module should always output logits; PyTorch loss
        # functions have numerically stable implementations with logits. During
        # inference, logits are converted to probabilities by applying sigmoid,
        # BUT OUTSIDE this module.
        #
        ######################################################################

        # Replace these lines with your code, keep variable names unchanged.
        self.pred_cls = None  # Class prediction conv
        self.pred_box = None  # Box regression conv
        self.pred_ctr = None  # Centerness conv

        # Replace "pass" statement with your code
        class_pred = nn.Conv2d(stem_channels[-1], num_classes, kernel_size=3, stride=1, padding=1)
        nn.init.normal(class_pred.weight)
        nn.init.zeros_(class_pred.bias)
        
        centerness_pred = nn.Conv2d(stem_channels[-1], 1, kernel_size=3, stride=1, padding=1)
        nn.init.normal(centerness_pred.weight)
        nn.init.zeros_(centerness_pred.bias)
        
        box_pred = nn.Conv2d(stem_channels[-1], 4, kernel_size=3, stride=1, padding=1)
        nn.init.normal(box_pred.weight)
        nn.init.zeros_(box_pred.bias)
        
        self.pred_cls = class_pred
        self.pred_box = box_pred
        self.pred_ctr = centerness_pred
        ######################################################################
        #                           END OF YOUR CODE                         #
        ######################################################################

        # OVERRIDE: Use a negative bias in `pred_cls` to improve training
        # stability. Without this, the training will most likely diverge.
        # STUDENTS: You do not need to get into details of why this is needed.
        torch.nn.init.constant_(self.pred_cls.bias, -math.log(99))

    def forward(self, feats_per_fpn_level: TensorDict) -> List[TensorDict]:
        """
        Accept FPN feature maps and predict the desired outputs at every location
        (as described above). Format them such that channels are placed at the
        last dimension, and (H, W) are flattened (having channels at last is
        convenient for computing loss as well as perforning inference).

        Args:
            feats_per_fpn_level: Features from FPN, keys {"p3", "p4", "p5"}. Each
                tensor will have shape `(batch_size, fpn_channels, H, W)`. For an
                input (224, 224) image, H = W are (28, 14, 7) for (p3, p4, p5).

        Returns:
            List of dictionaries, each having keys {"p3", "p4", "p5"}:
            1. Classification logits: `(batch_size, H * W, num_classes)`.
            2. Box regression deltas: `(batch_size, H * W, 4)`
            3. Centerness logits:     `(batch_size, H * W, 1)`
        """

        ######################################################################
        # TODO: Iterate over every FPN feature map and obtain predictions using
        # the layers defined above. Remember that prediction layers of box
        # regression and centerness will operate on output of `stem_box`,
        # and classification layer operates separately on `stem_cls`.
        #
        # CAUTION: The original FCOS model uses shared stem for centerness and
        # classification. Recent follow-up papers commonly place centerness and
        # box regression predictors with a shared stem, which we follow here.
        #
        # DO NOT apply sigmoid to classification and centerness logits.
        ######################################################################
        # Fill these with keys: {"p3", "p4", "p5"}, same as input dictionary.
        class_logits = {}
        boxreg_deltas = {}
        centerness_logits = {}

        # Replace "pass" statement with your code
        for key, val in feats_per_fpn_level.items():
            class_logits[key] = self.pred_cls(self.stem_cls(val))
            centerness_logits[key] = self.pred_ctr(self.stem_box(val))
            boxreg_deltas[key] = self.pred_box(self.stem_box(val))
        ######################################################################
        #                           END OF YOUR CODE                         #
        ######################################################################

        return [class_logits, boxreg_deltas, centerness_logits]
