from __future__ import print_function
import torch
from torch import nn, optim

from matplotlib import pyplot as plt
from a6_helper import show_images
import os


NOISE_DIM = 96


def hello_gan():
    print("Hello from gan.py!")


def sample_noise(batch_size, noise_dim, dtype=torch.float, device="cpu"):
    """
    Generate a PyTorch Tensor of uniform random noise.

    Input:
    - batch_size: Integer giving the batch size of noise to generate.
    - noise_dim: Integer giving the dimension of noise to generate.

    Output:
    - A PyTorch Tensor of shape (batch_size, noise_dim) containing uniform
      random noise in the range (-1, 1).
    """
    noise = None
    ##############################################################################
    # TODO: Implement sample_noise.                                              #
    ##############################################################################
    noise = torch.empty(size=(batch_size, noise_dim), dtype=dtype, device=device)
    noise = nn.init.xavier_uniform_(noise)

    return noise


def discriminator():
    """
    Build and return a PyTorch nn.Sequential model implementing the architecture
    in the notebook.
    """
    model = None
    ############################################################################
    # TODO: Implement discriminator.                                           #
    ############################################################################

    fc_1 = nn.Linear(784, 256)
    fc_2 = nn.Linear(256, 256)
    fc_3 = nn.Linear(256, 1)
    model = nn.Sequential(
        fc_1,
        nn.LeakyReLU(),
        fc_2,
        nn.LeakyReLU(),
        fc_3
    )

    return model


def generator(noise_dim=NOISE_DIM):
    """
    Build and return a PyTorch nn.Sequential model implementing the architecture
    in the notebook.
    """
    model = None
    ############################################################################
    # TODO: Implement generator.                                               #
    ############################################################################

    fc_1 = nn.Linear(noise_dim, 1024)
    fc_2 = nn.Linear(1024,1024)
    fc_3 = nn.Linear(1024,784)
    model = nn.Sequential(
        fc_1,
        nn.ReLU(),
        fc_2,
        nn.ReLU(),
        fc_3,
        nn.Tanh()
    )


    return model


def discriminator_loss(logits_real:torch.Tensor, logits_fake):
    """
    Computes the discriminator loss described above.

    Inputs:
    - logits_real: PyTorch Tensor of shape (N,) giving scores for the real data.
    - logits_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.

    Returns:
    - loss: PyTorch Tensor containing (scalar) the loss for the discriminator.
    """
    loss = None
    ##############################################################################
    # TODO: Implement discriminator_loss.                                        #
    ##############################################################################
  
    size = logits_fake.shape
    device = logits_real.device
    d_real_labels = torch.ones(size=size, device=device, dtype=torch.float)
    d_fake_labels = torch.zeros(size=size, device=device, dtype=torch.float)
    
    fake_label_loss = nn.functional.binary_cross_entropy_with_logits(logits_fake,  d_fake_labels)
    true_label_loss = nn.functional.binary_cross_entropy_with_logits(logits_real, d_real_labels)
    loss = fake_label_loss + true_label_loss
    
    return loss


def generator_loss(logits_fake:torch.Tensor):
    """
    Computes the generator loss described above.

    Inputs:
    - logits_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.

    Returns:
    - loss: PyTorch Tensor containing the (scalar) loss for the generator.
    """
    loss = None
    ##############################################################################
    # TODO: Implement generator_loss.                                            #
    ##############################################################################
    
    size = logits_fake.shape
    device = logits_fake.device
    d_real_labels = torch.ones(size=size, device=device, dtype=torch.float)
    loss = nn.functional.binary_cross_entropy_with_logits(logits_fake,  d_real_labels)
    
    return loss


def get_optimizer(model:nn.Module):
    """
    Construct and return an Adam optimizer for the model with learning rate 1e-3,
    beta1=0.5, and beta2=0.999.

    Input:
    - model: A PyTorch model that we want to optimize.

    Returns:
    - An Adam optimizer for the model with the desired hyperparameters.
    """
    optimizer = None
    ##############################################################################
    # TODO: Implement optimizer.                                                 #
    ##############################################################################
    
    optimizer = optim.Adam(model.parameters(),betas=(0.5, 0.999))
  
    return optimizer


def ls_discriminator_loss(scores_real, scores_fake):
    """
    Compute the Least-Squares GAN loss for the discriminator.

    Inputs:
    - scores_real: PyTorch Tensor of shape (N,) giving scores for the real data.
    - scores_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.

    Outputs:
    - loss: A PyTorch Tensor containing the loss.
    """
    loss = None
    ##############################################################################
    # TODO: Implement ls_discriminator_loss.                                     #
    ##############################################################################
    
    n = scores_real.shape[0]
    size = scores_real.shape
    device = scores_real.device

    least_real = scores_real - torch.ones(size, device=device)
    least_fake = scores_fake - torch.zeros(size, device=device)

    loss =  0.5*((least_real**2).sum() + (least_fake**2).sum()) / n
  
    return loss


def ls_generator_loss(scores_fake):
    """
    Computes the Least-Squares GAN loss for the generator.

    Inputs:
    - scores_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.

    Outputs:
    - loss: A PyTorch Tensor containing the loss.
    """
    loss = None
    ##############################################################################
    # TODO: Implement ls_generator_loss.                                         #
    ##############################################################################
    
    n = scores_fake.shape[0]
    size = scores_fake.shape
    device = scores_fake.device

    least_fake = scores_fake-torch.ones(size, device=device)
    loss = ((least_fake**2).sum() * 0.5) / n
  
    return loss


def build_dc_classifier():
    """
    Build and return a PyTorch nn.Sequential model for the DCGAN discriminator
    implementing the architecture in the notebook.
    """
    model = None
    ############################################################################
    # TODO: Implement build_dc_classifier.                                     #
    ############################################################################
    
    # (H - kernel_size + 2*padding) / stride + 1
    model = nn.Sequential(
        nn.Unflatten(dim=1,unflattened_size=[1,28,28]),
        nn.Conv2d(in_channels=1,out_channels=32,kernel_size=5, stride=1), #24
        nn.LeakyReLU(),
        nn.MaxPool2d(2,2), #12
        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1), #8
        nn.LeakyReLU(),
        nn.MaxPool2d(2,2), #4
        nn.Flatten(),
        nn.Linear(in_features=64*4*4, out_features=4*4*64),
        nn.LeakyReLU(),
        nn.Linear(in_features=4*4*64, out_features=1)
    )
    

    return model


def build_dc_generator(noise_dim=NOISE_DIM):
    """
    Build and return a PyTorch nn.Sequential model implementing the DCGAN
    generator using the architecture described in the notebook.
    """
    model = None
    ############################################################################
    # TODO: Implement build_dc_generator.                                      #
    ############################################################################
    
    model = nn.Sequential(
        nn.Linear(in_features=noise_dim, out_features=1024),
        nn.ReLU(inplace=True),
        nn.BatchNorm1d(num_features=1024),
        nn.Linear(in_features=1024, out_features=6272),
        nn.ReLU(inplace=True),
        nn.BatchNorm1d(num_features=6272),
        nn.Unflatten(dim=1, unflattened_size=[128, 7, 7]),
        nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1), #14
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(num_features=64),
        nn.ConvTranspose2d(in_channels=64, out_channels=1, kernel_size=4, stride=2, padding=1), #28
        nn.Tanh(),
        nn.Flatten()
    )
    

    return model




def run_a_gan(loader_train ,D, G, D_solver, G_solver, discriminator_loss, generator_loss, save_filename, show_every=250, 
              batch_size=128, noise_size=96, num_epochs=10, device):
  """
  Train a GAN!
  
  Inputs:
  - loader_train:
  - D, G: PyTorch models for the discriminator and generator
  - D_solver, G_solver: torch.optim Optimizers to use for training the
    discriminator and generator.
  - discriminator_loss, generator_loss: Functions to use for computing the generator and
    discriminator loss, respectively.
  - show_every: Show samples after every show_every iterations.
  - batch_size: Batch size to use for training.
  - noise_size: Dimension of the noise to use as input to the generator.
  - num_epochs: Number of epochs over the training dataset to use for training.
  - device: 
  """
  iter_count = 0
  for epoch in range(num_epochs):
    for x, _ in loader_train:
      if len(x) != batch_size:
        continue
      D_solver.zero_grad()
      real_data = x.view(-1, 784).to(device)
      logits_real = D(2* (real_data - 0.5))

      g_fake_seed = sample_noise(batch_size, noise_size, dtype=real_data.dtype, device=real_data.device)
      fake_images = G(g_fake_seed).detach()
      logits_fake = D(fake_images)

      d_total_error = discriminator_loss(logits_real, logits_fake)
      d_total_error.backward()        
      D_solver.step()

      G_solver.zero_grad()
      g_fake_seed = sample_noise(batch_size, noise_size, dtype=real_data.dtype, device=real_data.device)
      fake_images = G(g_fake_seed)

      gen_logits_fake = D(fake_images)
      g_error = generator_loss(gen_logits_fake)
      g_error.backward()
      G_solver.step()

      if (iter_count % show_every == 0):
        print('Iter: {}, D: {:.4}, G:{:.4}'.format(iter_count,d_total_error.item(),g_error.item()))
        imgs_numpy = fake_images.data.cpu()#.numpy()
        show_images(imgs_numpy[0:16])
        plt.show()
        print()
      iter_count += 1
    if epoch == num_epochs - 1:
      show_images(imgs_numpy[0:16])
      plt.savefig(os.path.join(".",save_filename))