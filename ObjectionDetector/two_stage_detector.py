
class RPNPredictionNetwork(nn.Module):
    """
    RPN prediction network that accepts FPN feature maps from different levels
    and makes two predictions for every anchor: objectness and box deltas.

    Faster R-CNN typically uses (p2, p3, p4, p5) feature maps. We will exclude
    p2 for have a small enough model for Colab.

    Conceptually this module is quite similar to `FCOSPredictionNetwork`.
    """

    def __init__(
        self, in_channels: int, stem_channels: List[int], num_anchors: int = 3
    ):
        """
        Args:
            in_channels: Number of channels in input feature maps. This value
                is same as the output channels of FPN.
            stem_channels: List of integers giving the number of output channels
                in each convolution layer of stem layers.
            num_anchors: Number of anchor boxes assumed per location (say, `A`).
                Faster R-CNN without an FPN uses `A = 9`, anchors with three
                different sizes and aspect ratios. With FPN, it is more common
                to have a fixed size dependent on the stride of FPN level, hence
                `A = 3` is default - with three aspect ratios.
        """
        super().__init__()

        self.num_anchors = num_anchors
        ######################################################################
        # TODO: Create a stem of alternating 3x3 convolution layers and RELU
        # activation modules. RPN shares this stem for objectness and box
        # regression (unlike FCOS, that uses separate stems).
        #
        # Use `in_channels` and `stem_channels` for creating these layers, the
        # docstring above tells you what they mean. Initialize weights of each
        # conv layer from a normal distribution with mean = 0 and std dev = 0.01
        # and all biases with zero. Use conv stride = 1 and zero padding such
        # that size of input features remains same: remember we need predictions
        # at every location in feature map, we shouldn't "lose" any locations.
        ######################################################################
        # Fill this list. It is okay to use your implementation from
        # `FCOSPredictionNetwork` for this code block.
        stem_rpn = []
        # Replace "pass" statement with your code
        for out_chanel in stem_channels:
            shared_conv = nn.Conv2d(in_channels, out_chanel, kernel_size=3, stride=1, padding=1)
            nn.init.normal_(shared_conv.weight, mean=0, std=0.01)
            nn.init.zeros_(shared_conv.bias)
            stem_rpn.append(shared_conv)
            stem_rpn.append(nn.ReLU())
            in_channels = out_chanel
        # Wrap the layers defined by student into a `nn.Sequential` module:
        self.stem_rpn = nn.Sequential(*stem_rpn)
        ######################################################################
        # TODO: Create TWO 1x1 conv layers for individually to predict
        # objectness and box deltas for every anchor, at every location.
        #
        # Objectness is obtained by applying sigmoid to its logits. However,
        # DO NOT initialize a sigmoid module here. PyTorch loss functions have
        # numerically stable implementations with logits.
        ######################################################################

        # Replace these lines with your code, keep variable names unchanged.
        self.pred_obj = None  # Objectness conv
        self.pred_box = None  # Box regression conv

        # Replace "pass" statement with your code
        object_conv = nn.Conv2d(stem_channels[-1], num_anchors, kernel_size=1, stride=1)
        nn.init.normal_(object_conv.weight)
        nn.init.zeros_(object_conv.bias)
        box_conv = nn.Conv2d(stem_channels[-1], num_anchors * 4, kernel_size=1, stride=1)
        nn.init.normal_(box_conv.weight)
        nn.init.zeros_(box_conv.bias)

        self.pred_obj = object_conv
        self.pred_box = box_conv
        ######################################################################
        #                           END OF YOUR CODE                         #
        ######################################################################

    def forward(self, feats_per_fpn_level: TensorDict) -> List[TensorDict]:
        """
        Accept FPN feature maps and predict desired quantities for every anchor
        at every location. Format the output tensors such that feature height,
        width, and number of anchors are collapsed into a single dimension (see
        description below in "Returns" section) this is convenient for computing
        loss and perforning inference.

        Args:
            feats_per_fpn_level: Features from FPN, keys {"p3", "p4", "p5"}.
                Each tensor will have shape `(batch_size, fpn_channels, H, W)`.

        Returns:
            List of dictionaries, each having keys {"p3", "p4", "p5"}:
            1. Objectness logits:     `(batch_size, H * W * num_anchors)`
            2. Box regression deltas: `(batch_size, H * W * num_anchors, 4)`
        """

        ######################################################################
        # TODO: Iterate over every FPN feature map and obtain predictions using
        # the layers defined above. DO NOT apply sigmoid to objectness logits.
        ######################################################################
        # Fill these with keys: {"p3", "p4", "p5"}, same as input dictionary.
        object_logits = {}
        boxreg_deltas = {}

        # Replace "pass" statement with your code
        for level_name, fpn_feats in feats_per_fpn_level.items():
            batch_size = fpn_feats.shape[0]
            stem_rpn = self.stem_rpn(fpn_feats)
            object_logits[level_name] = self.pred_obj(stem_rpn).view(batch_size, -1)
            pred_box = self.pred_box(stem_rpn)
            _,_,H, W = pred_box.shape
            pred_box = pred_box.reshape(batch_size, self.num_anchors * H * W, 4)
            boxreg_deltas[level_name] = pred_box
        ######################################################################
        #                           END OF YOUR CODE                         #
        ######################################################################

        return [object_logits, boxreg_deltas]

