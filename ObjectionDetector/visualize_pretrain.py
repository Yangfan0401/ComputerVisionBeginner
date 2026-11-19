from torchvision import transforms
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from eecs498.utils import detection_visualizer
from a4_helper import reset_seed, VOC2007DetectionTiny, train_detector, inference_with_detector
# Define an "inverse" transform for the image that un-normalizes by ImageNet color
# and mean. Without this, the images will NOT be visually understandable.


IMAGE_SHAPE = (224, 224)
train_dataset = VOC2007DetectionTiny(
    "VOC2007",
    "train",
    image_size=IMAGE_SHAPE[0],
    download=False,  # True (for the first time)
)
val_dataset = VOC2007DetectionTiny("VOC2007", "val", image_size=IMAGE_SHAPE[0])

inverse_norm = transforms.Compose(
    [
        transforms.Normalize(mean=[0., 0., 0.], std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
        transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1., 1., 1.]),
    ]
)

for idx, (_, image, gt_boxes) in enumerate(train_dataset):
    # Stop after visualizing three images.
    if idx > 2:
        break

    # Un-normalize image to bring in [0, 1] RGB range.
    image = inverse_norm(image)

    # Remove padded boxes from visualization.
    is_valid = gt_boxes[:, 4] >= 0
    detection_visualizer(image, val_dataset.idx_to_class, gt_boxes[is_valid])