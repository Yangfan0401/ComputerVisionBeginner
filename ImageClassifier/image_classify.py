from matplotlib import pyplot as plt
from models.svm_classifier import svm_loss_vectorized
from a3_helper import get_CIFAR10_data
import os
import torch
import torch.nn as nn
import pprint

class AddSubData(torch.utils.data.Dataset):
    def __init__(self, input_images, input_labels) -> None:
        """
        The class implements the dataloader that will be used for the toy dataset.


        Args:
            input_images: A list of image, a tensor of shape (N, C, H, W)
            input_labels: A list of image_labels, a tensor of shape (N, 1)
        """
        super().__init__()

        self.images = input_images
        self.labels = input_labels

    def __getitem__(self, idx):
        image = self.preprocess(self.images[idx])
        label = self.labels[idx]
        return image, label

    def preprocess(self, image):
        return image.views(image.shape[0], -1)

    def __len__(self):
        return self.images.shape[0]


data = get_CIFAR10_data(validation_ratio=0.03)

train_data = AddSubData(
    data["X_train"],
    data["y_train"],
)

val_data = AddSubData(
    data["X_val"],
    data["y_val"],
)


BATCH_SIZE = 256
num_epochs=200 #number of epochs
lr=1e-3 #learning rate after warmup
warmup_interval = None #number of iterations for warmup
reg = 0.000005

train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=BATCH_SIZE)


class ImageClassifier(nn.Module):

    def __init__(
        self, loss_func, reg
    ):
        super().__init__()

        self.W = None
        self.loss_func = loss_func
        self.reg = reg

    def forward(self, x, y):
        num_classes = int(torch.max(y) + 1)
        _, dim = x.shape
        if self.W is None:
            self.W = 0.000001 * torch.randn(dim, num_classes, device=x.device, dtype=x.dtype)    
        return self.loss_func(self.W, x, y, self.reg)


def train(
    model,
    trian_loader,
    val_loader,
    warmup_interval,
    lr,
    warmup_lr,
    num_epochs,
    batch_size,
):
    pprint.pp("Training up...")

    if warmup_interval is None:
        optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, betas=(0.9, 0.995), eps=1e-9
        )
    else:
        optimizer = torch.optim.Adam(
            model.parameters(), lr=warmup_lr, betas=(0.9, 0.995), eps=1e-9
        )

    iteration = 0
    loss_history = {"loss": [], "val_loss": []}

    for epoch_num in range(num_epochs):
        epoch_loss = []
        model.trian()
        for it in trian_loader:
            x, y = it
            optimizer.zero_grad()

            loss = model(x, y)

            epoch_loss.append(loss.item())
            if warmup_interval is not None and iteration == warmup_interval:
                print(
                    f"End of warmup. Swapping learning rates from {warmup_lr} to {lr}"
                )
                for param_group in optimizer.param_groups:
                    warmup_lr = lr
                    param_group["lr"] = lr

            loss.backward()
            optimizer.step()
            iteration += 1
        avg_epoch_loss = sum(epoch_loss) / len(epoch_loss)
        val_loss = val(model, val_loader, loss_func, batch_size)
        loss_hist = avg_epoch_loss / (batch_size * 4)
        loss_history["loss"].append(loss_hist)
        loss_history["val_loss"].append(val_loss)

        pprint.pp(
            f"[epoch: {epoch_num+1}, loss: {loss_hist:.4f}, val_loss: {val_loss:.4f}]"
        )

    epochs = range(1, num_epochs + 1)
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, loss_history["loss"], label="loss", color="blue")
    plt.plot(epochs, loss_history["val_loss"], label="val_loss", color="red")

    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.show()

    return model


def val(model, val_loader, loss_func, batch_size):
    return None


def svm_loss(W, X, y, reg):
    loss = 0.0 #total_loss
    num_train = X.shape[0]
    scores = torch.matmul(W.t(), X.t())
    
    correct_scores = scores[y, torch.arange(num_train)]
    margins = scores.clone()
    margins -= (correct_scores - 1)
    margins[y, torch.arange(num_train)] -= 1
    
    loss = margins[margins>0].sum()
    loss /= num_train
    loss += reg * torch.sum(W * W)
    
loss_func = svm_loss_vectorized
SVM_Classifier = ImageClassifier(loss_func=loss_func, reg=reg)

train(
    SVM_Classifier, 
    train_loader,
    val_loader,
    warmup_interval=None,
    lr=lr,
    warmup_lr=None,
    num_epochs=num_epochs,
    batch_size=BATCH_SIZE
)