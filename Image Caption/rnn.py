import os
import time
import math
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from rnn_lstm_captioning import (
    ImageEncoder,
    WordEmbedding,
    temporal_softmax_loss,
    CaptioningRNN
)
from a5_helper import load_coco_captions, decode_captions, train_captioner
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from eecs498.data import preprocess_cifar10
from eecs498.grad import reset_seed, rel_error, compute_numeric_gradient
from eecs498.solver import Solver
from eecs498.utils import detection_visualizer, attention_visualizer


os.environ["TZ"] = "US/Eastern"
time.tzset()
GOOGLE_DRIVE_PATH = "."
rnn_lstm_path = os.path.join(GOOGLE_DRIVE_PATH, "rnn_lstm_captioning.py")
rnn_lstm_edit_time = time.ctime(os.path.getmtime(rnn_lstm_path))
print("rnn_lstm_captioning.py last edited on %s" % rnn_lstm_edit_time)

plt.rcParams["figure.figsize"] = (15.0, 8.0)  # set default size of plots
plt.rcParams["font.size"] = 12
plt.rcParams["image.interpolation"] = "nearest"
plt.rcParams["image.cmap"] = "gray"


##############################################################################
# Recurrent Neural Network                                                   #
##############################################################################
def rnn_step_forward(x, prev_h, Wx, Wh, b):
    """
    Run the forward pass for a single timestep of a vanilla RNN that uses a tanh
    activation function.

    The input data has dimension D, the hidden state has dimension H, and we use
    a minibatch size of N.

    Args:
        x: Input data for this timestep, of shape (N, D).
        prev_h: Hidden state from previous timestep, of shape (N, H)
        Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
        Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
        b: Biases, of shape (H,)

    Returns a tuple of:
        next_h: Next hidden state, of shape (N, H)
        cache: Tuple of values needed for the backward pass.
    """
    next_h, cache = None, None
    ##########################################################################
    # TODO: Implement a single forward step for the vanilla RNN. Store next
    # hidden state and any values you need for the backward pass in the next_h
    # and cache variables respectively.
    ##########################################################################
    # Replace "pass" statement with your code
    next_h = x @ Wx + prev_h @ Wh + b
    next_h = F.tanh(next_h)
    cache = (x, Wx, prev_h, Wh, b, next_h)
    ##########################################################################
    #                             END OF YOUR CODE                           #
    ##########################################################################
    return next_h, cache


def rnn_step_backward(dnext_h, cache):
    """
    Backward pass for a single timestep of a vanilla RNN.

    Args:
        dnext_h: Gradient of loss with respect to next hidden state, of shape (N, H)
        cache: Cache object from the forward pass

    Returns a tuple of:
        dx: Gradients of input data, of shape (N, D)
        dprev_h: Gradients of previous hidden state, of shape (N, H)
        dWx: Gradients of input-to-hidden weights, of shape (D, H)
        dWh: Gradients of hidden-to-hidden weights, of shape (H, H)
        db: Gradients of bias vector, of shape (H,)
    """
    dx, dprev_h, dWx, dWh, db = None, None, None, None, None
    ##########################################################################
    # TODO: Implement the backward pass for a single step of a vanilla RNN.
    #
    # HINT: For the tanh function, you can compute the local derivative in
    # terms of the output value from tanh.
    ##########################################################################
    # Replace "pass" statement with your code
    x, Wx, prev_h, Wh, _, next_h = cache
    dt = dnext_h * (1 - next_h**2)

    dx = dt @ Wx.T
    dWx = x.T @ dt
    dprev_h = dt @ Wh.T
    dWh = prev_h.T @ dt
    db = dt.sum(dim=0)
    ##########################################################################
    #                             END OF YOUR CODE                           #
    ##########################################################################
    return dx, dprev_h, dWx, dWh, db


def rnn_forward(x, h0, Wx, Wh, b):
    """
    Run a vanilla RNN forward on an entire sequence of data. We assume an input
    sequence composed of T vectors, each of dimension D. The RNN uses a hidden
    size of H, and we work over a minibatch containing N sequences. After running
    the RNN forward, we return the hidden states for all timesteps.

    Args:
        x: Input data for the entire timeseries, of shape (N, T, D).
        h0: Initial hidden state, of shape (N, H)
        Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
        Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
        b: Biases, of shape (H,)

    Returns a tuple of:
        h: Hidden states for the entire timeseries, of shape (N, T, H).
        cache: Values needed in the backward pass
    """
    h, cache = None, None
    ##########################################################################
    # TODO: Implement forward pass for a vanilla RNN running on a sequence of
    # input data. You should use the rnn_step_forward function that you defined
    # above. You can use a for loop to help compute the forward pass.
    ##########################################################################
    # Replace "pass" statement with your code
    _, T, _ = x.shape

    prev_h = h0
    h = []
    cache = []

    for t in range(T):
        next_h, step_cache = rnn_step_forward(x[:, t, :], prev_h, Wx, Wh, b)

        h.append(next_h)
        cache.append(step_cache)

        prev_h = next_h

    h = torch.stack(h, dim=1)
    ##########################################################################
    #                             END OF YOUR CODE                           #
    ##########################################################################
    return h, cache


def rnn_backward(dh, cache):
    """
    Compute the backward pass for a vanilla RNN over an entire sequence of data.

    Args:
        dh: Upstream gradients of all hidden states, of shape (N, T, H).

    NOTE: 'dh' contains the upstream gradients produced by the
    individual loss functions at each timestep, *not* the gradients
    being passed between timesteps (which you'll have to compute yourself
    by calling rnn_step_backward in a loop).

    Returns a tuple of:
        dx: Gradient of inputs, of shape (N, T, D)
        dh0: Gradient of initial hidden state, of shape (N, H)
        dWx: Gradient of input-to-hidden weights, of shape (D, H)
        dWh: Gradient of hidden-to-hidden weights, of shape (H, H)
        db: Gradient of biases, of shape (H,)
    """
    dx, dh0, dWx, dWh, db = None, None, None, None, None
    ##########################################################################
    # TODO: Implement the backward pass for a vanilla RNN running an entire
    # sequence of data. You should use the rnn_step_backward function that you
    # defined above. You can use a for loop to help compute the backward pass.
    ##########################################################################
    # Replace "pass" statement with your code
    T = len(cache)

    N, _, H = dh.shape
    D = cache[0][0].shape[1]

    dx = torch.zeros(N, T, D)

    dWx = torch.zeros_like(cache[0][1])
    dWh = torch.zeros_like(cache[0][3])
    db = torch.zeros_like(cache[0][4])

    dprev_h = torch.zeros(N, H)

    for t in reversed(range(T)):
        dnext_h = dh[:, t, :] + dprev_h

        dx_t, dprev_h, dWx_t, dWh_t, db_t = rnn_step_backward(dnext_h, cache[t])

        dx[:, t, :] = dx_t

        dWx += dWx_t
        dWh += dWh_t
        db += db_t

    dh0 = dprev_h
    ##########################################################################
    #                             END OF YOUR CODE                           #
    ##########################################################################
    return dx, dh0, dWx, dWh, db


class RNN(nn.Module):
    """
    Single-layer vanilla RNN module.

    You don't have to implement anything here but it is highly recommended to
    read through the code as you will implement subsequent modules.
    """

    def __init__(self, input_dim: int, hidden_dim: int):
        """
        Initialize an RNN. Model parameters to initialize:
            Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
            Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
            b: Biases, of shape (H,)

        Args:
            input_dim: Input size, denoted as D before
            hidden_dim: Hidden size, denoted as H before
        """
        super().__init__()

        # Register parameters
        self.Wx = nn.Parameter(
            torch.randn(input_dim, hidden_dim).div(math.sqrt(input_dim))
        )
        self.Wh = nn.Parameter(
            torch.randn(hidden_dim, hidden_dim).div(math.sqrt(hidden_dim))
        )
        self.b = nn.Parameter(torch.zeros(hidden_dim))

    def forward(self, x, h0):
        """
        Args:
            x: Input data for the entire timeseries, of shape (N, T, D)
            h0: Initial hidden state, of shape (N, H)

        Returns:
            hn: The hidden state output
        """
        hn, _ = rnn_forward(x, h0, self.Wx, self.Wh, self.b)
        return hn

    def step_forward(self, x, prev_h):
        """
        Args:
            x: Input data for one time step, of shape (N, D)
            prev_h: The previous hidden state, of shape (N, H)

        Returns:
            next_h: The next hidden state, of shape (N, H)
        """
        next_h, _ = rnn_step_forward(x, prev_h, self.Wx, self.Wh, self.b)
        return next_h



DEVICE = "cpu"
to_float = {"dtype": torch.float32, "device": DEVICE}
to_double = {"dtype": torch.float64, "device": DEVICE}

# Set a few constants related to data loading.
IMAGE_SHAPE = (112, 112)
NUM_WORKERS = 1

# Batch size used for full training runs:
BATCH_SIZE = 256

# Batch size used for overfitting sanity checks:
OVR_BATCH_SIZE = BATCH_SIZE // 8

# Batch size used for visualization:
VIS_BATCH_SIZE = 4

# Download and load serialized COCO data from coco.pt
# It contains a dictionary of
# "train_images" - resized training images (IMAGE_SHAPE)
# "val_images" - resized validation images (IMAGE_SHAPE)
# "train_captions" - tokenized and numericalized training captions
# "val_captions" - tokenized and numericalized validation captions
# "vocab" - caption vocabulary, including "idx_to_token" and "token_to_idx"

if os.path.isfile("./datasets/coco.pt"):
    print("COCO data exists!")
else:
    print("downloading COCO dataset")

# load COCO data from coco.pt, loaf_COCO is implemented in a5_helper.py
data_dict = load_coco_captions(path="./datasets/coco.pt")

num_train = data_dict["train_images"].size(0)
num_val = data_dict["val_images"].size(0)

# declare variables for special tokens
NULL_index = data_dict["vocab"]["token_to_idx"]["<NULL>"]
START_index = data_dict["vocab"]["token_to_idx"]["<START>"]
END_index = data_dict["vocab"]["token_to_idx"]["<END>"]
UNK_index = data_dict["vocab"]["token_to_idx"]["<UNK>"]

# # # Sample a minibatch and show the reshaped 112x112 images and captions
# # sample_idx = torch.randint(0, num_train, (VIS_BATCH_SIZE, ))
# # sample_images = data_dict["train_images"][sample_idx]
# # sample_captions = data_dict["train_captions"][sample_idx]
# # for i in range(VIS_BATCH_SIZE):
# #     plt.imshow(sample_images[i].permute(1, 2, 0))
# #     plt.axis("off")
# #     caption_str = decode_captions(
# #         sample_captions[i], data_dict["vocab"]["idx_to_token"]
# #     )
# #     plt.title(caption_str)
# #     plt.tight_layout()
# #     plt.show()
#
# # NOTE: 搭建RNN前向传播函数
# N, D, H = 3, 10, 4
#
# x = torch.linspace(-0.4, 0.7, steps=N * D, **to_double).view(N, D)
# prev_h = torch.linspace(-0.2, 0.5, steps=N * H, **to_double).view(N, H)
# Wx = torch.linspace(-0.1, 0.9, steps=D * H, **to_double).view(D, H)
# Wh = torch.linspace(-0.3, 0.7, steps=H * H, **to_double).view(H, H)
# b = torch.linspace(-0.2, 0.4, steps=H, **to_double)
#
#
# next_h, _ = rnn_step_forward(x, prev_h, Wx, Wh, b)
# expected_next_h = torch.tensor(
#     [
#         [-0.58172089, -0.50182032, -0.41232771, -0.31410098],
#         [0.66854692, 0.79562378, 0.87755553, 0.92795967],
#         [0.97934501, 0.99144213, 0.99646691, 0.99854353],
#     ],
#     **to_double,
# )
#
# print("next_h error: ", rel_error(expected_next_h, next_h))
#
#
# # NOTE: 搭建RNN反向传播函数
# reset_seed(0)
#
# N, D, H = 4, 5, 6
# x = torch.randn(N, D, **to_double)
# h = torch.randn(N, H, **to_double)
# Wx = torch.randn(D, H, **to_double)
# Wh = torch.randn(H, H, **to_double)
# b = torch.randn(H, **to_double)
#
# out, cache = rnn_step_forward(x, h, Wx, Wh, b)
#
# dnext_h = torch.randn(*out.shape, **to_double)
#
# fx = lambda x: rnn_step_forward(x, h, Wx, Wh, b)[0]
# fh = lambda h: rnn_step_forward(x, h, Wx, Wh, b)[0]
# fWx = lambda Wx: rnn_step_forward(x, h, Wx, Wh, b)[0]
# fWh = lambda Wh: rnn_step_forward(x, h, Wx, Wh, b)[0]
# fb = lambda b: rnn_step_forward(x, h, Wx, Wh, b)[0]
#
# dx_num = compute_numeric_gradient(fx, x, dnext_h)
# dprev_h_num = compute_numeric_gradient(fh, h, dnext_h)
# dWx_num = compute_numeric_gradient(fWx, Wx, dnext_h)
# dWh_num = compute_numeric_gradient(fWh, Wh, dnext_h)
# db_num = compute_numeric_gradient(fb, b, dnext_h)
#
# # YOUR_TURN: Implement rnn_step_backward
# dx, dprev_h, dWx, dWh, db = rnn_step_backward(dnext_h, cache)
#
# print("dx error: ", rel_error(dx_num, dx))
# print("dprev_h error: ", rel_error(dprev_h_num, dprev_h))
# print("dWx error: ", rel_error(dWx_num, dWx))
# print("dWh error: ", rel_error(dWh_num, dWh))
# print("db error: ", rel_error(db_num, db))
#
#
# N, T, D, H = 2, 3, 4, 5
#
# x = torch.linspace(-0.1, 0.3, steps=N * T * D, **to_double).view(N, T, D)
# h0 = torch.linspace(-0.3, 0.1, steps=N * H, **to_double).view(N, H)
# Wx = torch.linspace(-0.2, 0.4, steps=D * H, **to_double).view(D, H)
# Wh = torch.linspace(-0.4, 0.1, steps=H * H, **to_double).view(H, H)
# b = torch.linspace(-0.7, 0.1, steps=H, **to_double)
#
# # YOUR_TURN: Implement rnn_forward
# h, _ = rnn_forward(x, h0, Wx, Wh, b)
# expected_h = torch.tensor(
#     [
#         [
#             [-0.42070749, -0.27279261, -0.11074945, 0.05740409, 0.22236251],
#             [-0.39525808, -0.22554661, -0.0409454, 0.14649412, 0.32397316],
#             [-0.42305111, -0.24223728, -0.04287027, 0.15997045, 0.35014525],
#         ],
#         [
#             [-0.55857474, -0.39065825, -0.19198182, 0.02378408, 0.23735671],
#             [-0.27150199, -0.07088804, 0.13562939, 0.33099728, 0.50158768],
#             [-0.51014825, -0.30524429, -0.06755202, 0.17806392, 0.40333043],
#         ],
#     ],
#     **to_double,
# )
# print("h error: ", rel_error(expected_h, h))
#
#
#
# reset_seed(0)
#
# N, D, T, H = 2, 3, 10, 5
#
# x = torch.randn(N, T, D, **to_double)
# h0 = torch.randn(N, H, **to_double)
# Wx = torch.randn(D, H, **to_double)
# Wh = torch.randn(H, H, **to_double)
# b = torch.randn(H, **to_double)
#
# out, cache = rnn_forward(x, h0, Wx, Wh, b)
#
# dout = torch.randn(*out.shape, **to_double)
#
# # YOUR_TURN: Implement rnn_backward
# dx, dh0, dWx, dWh, db = rnn_backward(dout, cache)
#
# fx = lambda x: rnn_forward(x, h0, Wx, Wh, b)[0]
# fh0 = lambda h0: rnn_forward(x, h0, Wx, Wh, b)[0]
# fWx = lambda Wx: rnn_forward(x, h0, Wx, Wh, b)[0]
# fWh = lambda Wh: rnn_forward(x, h0, Wx, Wh, b)[0]
# fb = lambda b: rnn_forward(x, h0, Wx, Wh, b)[0]
#
# dx_num = compute_numeric_gradient(fx, x, dout)
# dh0_num = compute_numeric_gradient(fh0, h0, dout)
# dWx_num = compute_numeric_gradient(fWx, Wx, dout)
# dWh_num = compute_numeric_gradient(fWh, Wh, dout)
# db_num = compute_numeric_gradient(fb, b, dout)
#
# print("dx error: ", rel_error(dx_num, dx))
# print("dh0 error: ", rel_error(dh0_num, dh0))
# print("dWx error: ", rel_error(dWx_num, dWx))
# print("dWh error: ", rel_error(dWh_num, dWh))
# print("db error: ", rel_error(db_num, db))
#
#
#
# reset_seed(0)
#
# N, D, T, H = 2, 3, 10, 5
#
# # set requires_grad=True
# x = torch.randn(N, T, D, **to_double, requires_grad=True)
# h0 = torch.randn(N, H, **to_double, requires_grad=True)
# Wx = torch.randn(D, H, **to_double, requires_grad=True)
# Wh = torch.randn(H, H, **to_double, requires_grad=True)
# b = torch.randn(H, **to_double, requires_grad=True)
#
# out, cache = rnn_forward(x, h0, Wx, Wh, b)
#
# dout = torch.randn(*out.shape, **to_double)
#
# # Manual backward:
# with torch.no_grad():
#     dx, dh0, dWx, dWh, db = rnn_backward(dout, cache)
#
# # Backward with autograd: the magic happens here!
# out.backward(dout)
#
# dx_auto, dh0_auto, dWx_auto, dWh_auto, db_auto = (
#     x.grad,
#     h0.grad,
#     Wx.grad,
#     Wh.grad,
#     b.grad,
# )
#
# print("dx error: ", rel_error(dx_auto, dx))
# print("dh0 error: ", rel_error(dh0_auto, dh0))
# print("dWx error: ", rel_error(dWx_auto, dWx))
# print("dWh error: ", rel_error(dWh_auto, dWh))
# print("db error: ", rel_error(db_auto, db))
#
#
# N, D, T, H = 2, 3, 10, 5
#
# x = torch.randn(N, T, D, **to_double)
# h0 = torch.randn(N, H, **to_double)
#
# rnn_module = RNN(D, H).to(**to_double)
#
# # Call forward in module:
# hn1 = rnn_module(x, h0)
#
# # Call without module: (but access weights from module)
# # Equivalent to above, we won't do this henceforth.
# Wx, Wh, b = rnn_module.Wx, rnn_module.Wh, rnn_module.b
# hn2, _ = rnn_forward(x, h0, Wx, Wh, b)
#
# print("Output error with/without module: ", rel_error(hn1, hn2))
#
# model = ImageEncoder(pretrained=True, verbose=True).to(device=DEVICE)

# N, T, V, D = 2, 4, 5, 3

# x = torch.tensor([[0, 3, 1, 2], [2, 1, 0, 3]]).long()
# W = torch.linspace(0, 1, steps=V * D, **to_double).view(V, D)

# # Copy custom weight vector for sanity check:
# model_emb = WordEmbedding(V, D).to(**to_double)
# model_emb.W_embed.data.copy_(W)
# out = model_emb(x)
# expected_out = torch.tensor(
#     [
#         [
#             [0.0, 0.07142857, 0.14285714],
#             [0.64285714, 0.71428571, 0.78571429],
#             [0.21428571, 0.28571429, 0.35714286],
#             [0.42857143, 0.5, 0.57142857],
#         ],
#         [
#             [0.42857143, 0.5, 0.57142857],
#             [0.21428571, 0.28571429, 0.35714286],
#             [0.0, 0.07142857, 0.14285714],
#             [0.64285714, 0.71428571, 0.78571429],
#         ],
#     ],
#     **to_double
# )

# print("out error: ", rel_error(expected_out, out))
#
#
# def check_loss(N, T, V, p):
#     x = 0.001 * torch.randn(N, T, V)
#     y = torch.randint(V, size=(N, T))
#     mask = torch.rand(N, T)
#     y[mask > p] = 0
#
#     # YOUR_TURN: Implement temporal_softmax_loss
#     print(temporal_softmax_loss(x, y, NULL_index).item())
#
#
# check_loss(1000, 1, 10, 1.0)  # Should be about 2.00-2.11
# check_loss(1000, 10, 10, 1.0)  # Should be about 20.6-21.0
# check_loss(5000, 10, 10, 0.1)  # Should be about 2.00-2.11
#
#

# reset_seed(0)

# N, D, W, H = 10, 400, 30, 40
# word_to_idx = {"<NULL>": 0, "cat": 2, "dog": 3}
# V = len(word_to_idx)
# T = 13

# model = CaptioningRNN(
#     word_to_idx,
#     input_dim=D,
#     wordvec_dim=W,
#     hidden_dim=H,
#     cell_type="rnn",
#     ignore_index=NULL_index,
# )
# # Copy parameters for sanity check:
# for k, v in model.named_parameters():
#     v.data.copy_(torch.linspace(-1.4, 1.3, steps=v.numel()).view(*v.shape))

# images = torch.randn(N, 3, *IMAGE_SHAPE)
# captions = (torch.arange(N * T) % V).view(N, T)

# loss = model(images, captions)
# expected_loss = 150.6090393066

# print("loss: ", loss)
# print("expected loss: ", expected_loss)
# print("difference: ", rel_error(torch.tensor(loss), torch.tensor(expected_loss)))


#
#
#
# reset_seed(0)
#
# # data input
# small_num_train = 50
# sample_idx = torch.linspace(0, num_train - 1, steps=small_num_train).long()
# small_image_data = data_dict["train_images"][sample_idx]
# small_caption_data = data_dict["train_captions"][sample_idx]
#
# # optimization arguments
# num_epochs = 80
#
# # create the image captioning model
# model = CaptioningRNN(
#     cell_type="rnn",
#     word_to_idx=data_dict["vocab"]["token_to_idx"],
#     input_dim=400,  # hard-coded, do not modify
#     hidden_dim=512,
#     wordvec_dim=256,
#     ignore_index=NULL_index,
# )
# model = model.to(**to_float)
#
# for learning_rate in [1e-3]:
#     print("learning rate is: ", learning_rate)
#     rnn_overfit, _ = train_captioner(
#         model,
#         small_image_data,
#         small_caption_data,
#         num_epochs=num_epochs,
#         batch_size=OVR_BATCH_SIZE,
#         learning_rate=learning_rate,
#         device=DEVICE,
#     )
#
#
#
# reset_seed(0)
#
# # data input
# small_num_train = num_train
# sample_idx = torch.randint(num_train, size=(small_num_train,))
# small_image_data = data_dict["train_images"][sample_idx]
# small_caption_data = data_dict["train_captions"][sample_idx]
#
#
# # create the image captioning model
# rnn_model = CaptioningRNN(
#     cell_type="rnn",
#     word_to_idx=data_dict["vocab"]["token_to_idx"],
#     input_dim=400,  # hard-coded, do not modify
#     hidden_dim=512,
#     wordvec_dim=256,
#     ignore_index=NULL_index,
# )
#
# for learning_rate in [1e-3]:
#     print("learning rate is: ", learning_rate)
#     rnn_model_submit, rnn_loss_submit = train_captioner(
#         rnn_model,
#         small_image_data,
#         small_caption_data,
#         num_epochs=60,
#         batch_size=BATCH_SIZE,
#         learning_rate=learning_rate,
#         device=DEVICE,
#     )
#
#
# rnn_model.eval()
#
# for split in ["train", "val"]:
#     sample_idx = torch.randint(
#         0, num_train if split == "train" else num_val, (VIS_BATCH_SIZE,)
#     )
#     sample_images = data_dict[split + "_images"][sample_idx]
#     sample_captions = data_dict[split + "_captions"][sample_idx]
#
#     # decode_captions is loaded from a5_helper.py
#     gt_captions = decode_captions(sample_captions, data_dict["vocab"]["idx_to_token"])
#
#     generated_captions = rnn_model.sample(sample_images.to(DEVICE))
#     generated_captions = decode_captions(
#         generated_captions, data_dict["vocab"]["idx_to_token"]
#     )
#
#     for i in range(VIS_BATCH_SIZE):
#         plt.imshow(sample_images[i].permute(1, 2, 0))
#         plt.axis("off")
#         plt.title(
#             f"[{split}] RNN Generated: {generated_captions[i]}\nGT: {gt_captions[i]}"
#         )
#         plt.show()
#
#
#
