import multiprocessing
import torch
from a4_helper import reset_seed, VOC2007DetectionTiny, train_detector
from models.fcos import FCOS
from torchvision import transforms

reset_seed(0)

# Set a few constants related to data loading.
NUM_CLASSES = 20
BATCH_SIZE = 16
IMAGE_SHAPE = (224, 224)
NUM_WORKERS = multiprocessing.cpu_count()
DEVICE = "cuda"


train_dataset = VOC2007DetectionTiny(
    "VOC2007",
    "train",
    image_size=IMAGE_SHAPE[0],
    download=False,  # True (for the first time)
)
val_dataset = VOC2007DetectionTiny("VOC2007", "val", image_size=IMAGE_SHAPE[0])


inverse_norm = transforms.Compose(
    [
        transforms.Normalize(
            mean=[0.0, 0.0, 0.0], std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
        ),
        transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1.0, 1.0, 1.0]),
    ]
)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE)

val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE)

small_dataset = torch.utils.data.Subset(
    train_dataset,
    torch.linspace(0, len(train_dataset) - 1, steps=BATCH_SIZE * 10).long(),
)
small_train_loader = torch.utils.data.DataLoader(small_dataset, batch_size=BATCH_SIZE)

# Take equally spaced examples from training dataset to make a subset.
# Small data overfitting
detector = FCOS(
    num_classes=NUM_CLASSES,
    fpn_channels=64,
    stem_channels=[64, 64],
)
detector = detector.to(DEVICE)

train_detector(
    detector,
    small_train_loader,
    learning_rate=1e-4,
    max_iters=500,
    log_period=20,
    device=DEVICE,
)
