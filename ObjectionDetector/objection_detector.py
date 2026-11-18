import multiprocessing
import torch
from a4_helper import reset_seed, VOC2007DetectionTiny, train_detector
from models.fcos import FCOS
import pprint
import time
import os
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

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=4, pin_memory=True)

val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=4, pin_memory=True)


# # Slightly larger detector than in above cell.
detector = FCOS(
    num_classes=NUM_CLASSES,
    fpn_channels=128,
    stem_channels=[128, 128],
)
detector = detector.to(DEVICE, dtype=torch.float)

start_time = time.time()
train_detector(
    detector,
    train_loader,
    learning_rate=1e-4,
    max_iters=9000,
    log_period=100,
    device=DEVICE,
)
total_time = time.time() - start_time
# 时分秒格式化
h = int(total_time // 3600)
m = int((total_time % 3600) // 60)
s = total_time % 60
pprint.pp(f"\n总训练时长: {h}h {m}m {s:.2f}s")
# After you've trained your model, save the weights for submission.
weights_path = os.path.join("checkpoints", "fcos_detector.pt")
torch.save(detector.state_dict(), weights_path)