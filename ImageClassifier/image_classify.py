from a3_helper import get_CIFAR10_data
import os
import torch
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

    def __getitem__(self, idx) :
        image = self.preprocess(self.images[idx])
        label = self.labels[idx]
        return image, label

    def preprocess(self, image):
        return image.views(image.shape[0], -1)

    def __len__(self):
        return self.images.shape[0]

data = get_CIFAR10_data(validation_ratio=0.03)

train_data = AddSubData(
    data["x_train"], 
    data["y_train"],
)

val_data = AddSubData(
    data["x_val"], 
    data["y_val"],
)


pprint.pp(train_data[0])


train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=BATCH_SIZE)



def train(model, 
    train_loader,
    val_loader,
    loss_func,
    num_epochs,
    learning_rates,
    weight_decacy,
    batch_size,):
    return None

    for epoch in range(num_epochs):
        model.train()
        

# train_model = train(
#     model
#     train_loader,
#     val_loader,
#     loss_func,
#     num_epochs,
#     lr,
#     weight_decacy,
#     batch_size,
# )
