import json
import math
import time
import torch.nn.functional as F
import torch.nn as nn
import torchvision
from sklearn import feature_extraction
import torch
import matplotlib.pyplot as plt
import seaborn
import torch


def load_coco_captions(path: str = "./datasets/coco.pt"):
    """
    Download and load serialized COCO data from coco.pt
    It contains a dictionary of
    "train_images" - resized training images (112x112)
    "val_images" - resized validation images (112x112)
    "train_captions" - tokenized and numericalized training captions
    "val_captions" - tokenized and numericalized validation captions
    "vocab" - caption vocabulary, including "idx_to_token" and "token_to_idx"

    Returns: a data dictionary
  """
    data_dict = torch.load(path)
    # print out all the keys and values from the data dictionary
    for k, v in data_dict.items():
        if type(v) == torch.Tensor:
            print(k, type(v), v.shape, v.dtype)
        else:
            print(k, type(v), v.keys())

    assert data_dict["train_images"].size(0) == data_dict["train_captions"].size(
        0
    ) and data_dict["val_images"].size(0) == data_dict["val_captions"].size(
        0
    ), "shapes of data mismatch!"

    print("\nTrain images shape: ", data_dict["train_images"].shape)
    print("Train caption tokens shape: ", data_dict["train_captions"].shape)
    print("Validation images shape: ", data_dict["val_images"].shape)
    print("Validation caption tokens shape: ", data_dict["val_captions"].shape)
    print(
        "total number of caption tokens: ", len(data_dict["vocab"]["idx_to_token"])
    )
    print(
        "mappings (list) from index to caption token: ",
        data_dict["vocab"]["idx_to_token"],
    )
    print(
        "mappings (dict) from caption token to index: ",
        data_dict["vocab"]["token_to_idx"],
    )

    return data_dict


def get_toy_data(path: str = "final_data.json"):
    return json.load(open(path))

class WordEmbedding(nn.Module):
    """
    Simplified version of torch.nn.Embedding.

    We operate on minibatches of size N where
    each sequence has length T. We assume a vocabulary of V words, assigning each
    word to a vector of dimension D.

    Args:
        x: Integer array of shape (N, T) giving ndices of words. Each element idx
      of x muxt be in the range 0 <= idx < V.i

    Returns a tuple of:
        out: Array of shape (N, T, D) giving word vectors for all input words.
    """

    def __init__(self, vocab_size: int, embed_size: int):
        super().__init__()

        # Register parameters
        self.W_embed = nn.Parameter(
            torch.randn(vocab_size, embed_size).div(math.sqrt(vocab_size))
        )

    def forward(self, x):
        out = None
        ######################################################################
        # TODO: Implement the forward pass for word embeddings.
        ######################################################################
        # Replace "pass" statement with your code
        N, T = x.shape
        D = self.W_embed.shape[1]
        out = torch.zeros(size=[N, T, D])
        for i in range(N):
            out[i, :, :] = self.W_embed[x[i], :]
        ######################################################################
        #                           END OF YOUR CODE                         #
        ######################################################################
        return out

class ImageEncoder(nn.Module):
    """
    Convolutional network that accepts images as input and outputs their spatial
    grid features. This module servesx as the image encoder in image captioning
    model. We will use a tiny RegNet-X 400MF model that is initialized with
    ImageNet-pretrained weights from Torchvision library.

    NOTE: We could use any convolutional network architecture, but we opt for a
    tiny RegNet model so it can train decently with a single K80 Colab GPU.
    """

    def __init__(self, pretrained: bool = True, verbose: bool = True):
        """
        Args:
            pretrained: Whether to initialize this model with pretrained weights
                from Torchvision library.
            verbose: Whether to log expected output shapes during instantiation.
        """
        super().__init__()
        self.cnn = torchvision.models.regnet_x_400mf(pretrained=pretrained)

        # Torchvision models return global average pooled features by default.
        # Our attention-based models may require spatial grid features. So we
        # wrap the ConvNet with torchvision's feature extractor. We will get
        # the spatial features right before the final classification layer.
        self.backbone = feature_extraction.create_feature_extractor(
            self.cnn, return_nodes={"trunk_output.block4": "c5"}
        )
        # We call these features "c5", a name that may sound familiar from the
        # object detection assignment. :-)

        # Pass a dummy batch of input images to infer output shape.
        dummy_out = self.backbone(torch.randn(2, 3, 224, 224))["c5"]
        self._out_channels = dummy_out.shape[1]

        if verbose:
            print("For input images in NCHW format, shape (2, 3, 224, 224)")
            print(f"Shape of output c5 features: {dummy_out.shape}")

        # Input image batches are expected to be float tensors in range [0, 1].
        # However, the backbone here expects these tensors to be normalized by
        # ImageNet color mean/std (as it was trained that way).
        # We define a function to transform the input images before extraction:
        self.normalize = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    @property
    def out_channels(self):
        """
        Number of output channels in extracted image features. You may access
        this value freely to define more modules to go with this encoder.
        """
        return self._out_channels

    def forward(self, images: torch.Tensor):
        # Input images may be uint8 tensors in [0-255], change them to float
        # tensors in [0-1]. Get float type from backbone (could be float32/64).
        if images.dtype == torch.uint8:
            images = images.to(dtype=self.cnn.stem[0].weight.dtype)
            images /= 255.0

        # Normalize images by ImageNet color mean/std.
        images = self.normalize(images)

        # Extract c5 features from encoder (backbone) and return.
        # shape: (B, out_channels, H / 32, W / 32)
        features = self.backbone(images)["c5"]
        return features


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
    dt = dnext_h * (1 - next_h ** 2)

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



def dot_product_attention(prev_h, A):
    """
    A simple scaled dot-product attention layer.

    Args:
        prev_h: The LSTM hidden state from previous time step, of shape (N, H)
        A: **Projected** CNN feature activation, of shape (N, H, 4, 4),
         where H is the LSTM hidden state size

    Returns:
        attn: Attention embedding output, of shape (N, H)
        attn_weights: Attention weights, of shape (N, 4, 4)

    """
    N, H, D_a, _ = A.shape

    attn, attn_weights = None, None
    ##########################################################################
    # TODO: Implement the scaled dot-product attention we described earlier. #
    # You will use this function for `AttentionLSTM` forward and sample      #
    # functions. HINT: Make sure you reshape attn_weights back to (N, 4, 4)! #
    ##########################################################################
    # Replace "pass" statement with your code
    flatten_A = A.reshape(N, H, -1)  # N, H, 16
    squeeze_prev_h = prev_h.unsqueeze(dim=1)  # N, 1, H
    attn_weights = (squeeze_prev_h @ flatten_A) / math.sqrt(D_a)  # N, 1, 16
    attn_weights = attn_weights.squeeze(dim=1)
    attn_weights = F.softmax(attn_weights, dim=1)  # N, 16

    squeeze_attn_weights = attn_weights.unsqueeze(dim=2)  # N, 16, 1
    attn = flatten_A @ squeeze_attn_weights
    attn = attn.reshape([N, H])
    attn_weights = attn_weights.reshape([N, D_a, D_a])
    ##########################################################################
    #                             END OF YOUR CODE                           #
    ##########################################################################

    return attn, attn_weights


def temporal_softmax_loss(x, y, ignore_index=None):
    """
    A temporal version of softmax loss for use in RNNs. We assume that we are
    making predictions over a vocabulary of size V for each timestep of a
    timeseries of length T, over a minibatch of size N. The input x gives scores
    for all vocabulary elements at all timesteps, and y gives the indices of the
    ground-truth element at each timestep. We use a cross-entropy loss at each
    timestep, *summing* the loss over all timesteps and *averaging* across the
    minibatch.

    As an additional complication, we may want to ignore the model output at some
    timesteps, since sequences of different length may have been combined into a
    minibatch and padded with NULL tokens. The optional ignore_index argument
    tells us which elements in the caption should not contribute to the loss.

    Args:
        x: Input scores, of shape (N, T, V)
        y: Ground-truth indices, of shape (N, T) where each element is in the
            range 0 <= y[i, t] < V

    Returns a tuple of:
        loss: Scalar giving loss
    """
    loss = None

    ##########################################################################
    # TODO: Implement the temporal softmax loss function.
    #
    # REQUIREMENT: This part MUST be done in one single line of code!
    #
    # HINT: Look up the function torch.functional.cross_entropy, set
    # ignore_index to the variable ignore_index (i.e., index of NULL) and
    # set reduction to either 'sum' or 'mean' (avoid using 'none' for now).
    #
    # We use a cross-entropy loss at each timestep, *summing* the loss over
    # all timesteps and *averaging* across the minibatch.
    ##########################################################################
    # Replace "pass" statement with your code
    loss = (
            F.cross_entropy(
                x.reshape(-1, x.shape[-1]),
                y.reshape(-1),
                ignore_index=ignore_index,
                reduction="sum",
            )
            / x.shape[0]
    )
    ##########################################################################
    #                             END OF YOUR CODE                           #
    ##########################################################################

    return loss


def train_captioner(
    model,
    image_data,
    caption_data,
    num_epochs,
    batch_size,
    learning_rate,
    lr_decay=1,
    device: torch.device = torch.device("cpu"),
):
    """
    Run optimization to train the model.
    """

    model = model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), learning_rate
    )
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda epoch: lr_decay ** epoch
    )

    # sample minibatch data
    iter_per_epoch = math.ceil(image_data.shape[0] // batch_size)
    loss_history = []

    for i in range(num_epochs):
        start_t = time.time()
        for j in range(iter_per_epoch):
            images, captions = (
                image_data[j * batch_size : (j + 1) * batch_size],
                caption_data[j * batch_size : (j + 1) * batch_size],
            )
            images = images.to(device)
            captions = captions.to(device)

            loss = model(images, captions)
            optimizer.zero_grad()
            loss.backward()
            loss_history.append(loss.item())
            optimizer.step()
        end_t = time.time()

        print(
            "(Epoch {} / {}) loss: {:.4f} time per epoch: {:.1f}s".format(
                i, num_epochs, loss.item(), end_t - start_t
            )
        )

        lr_scheduler.step()

    # plot the training losses
    plt.plot(loss_history)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Training loss history")
    plt.show()
    return model, loss_history


def decode_captions(captions, idx_to_word):
    """
    Decoding caption indexes into words.

    Args:
        captions: Caption indexes in a tensor of shape (N, T).
        idx_to_word: Mapping from the vocab index to word.

    Returns:
        decoded: A sentence (or a list of N sentences).
    """
    singleton = captions.ndim == 1
    captions = captions[None] if singleton else captions

    decoded = []
    N, T = captions.shape
    for i in range(N):
        words = []
        for t in range(T):
            word = idx_to_word[captions[i, t]]
            if word != "<NULL>":
                words.append(word)
            if word == "<END>":
                break
        decoded.append(" ".join(words))

    if singleton:
        decoded = decoded[0]
    return decoded


def train(
    model,
    train_dataloader,
    val_dataloader,
    loss_func,
    num_epochs,
    batch_size=32,
    warmup_lr=6e-6,
    warmup_interval=1000,
    lr=6e-4,
    device=torch.device("cpu"),
):
    print("Training started...")
    if warmup_interval is None:
        optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, betas=(0.9, 0.995), eps=1e-9
        )
    else:
        optimizer = torch.optim.Adam(
            model.parameters(), lr=warmup_lr, betas=(0.9, 0.995), eps=1e-9
        )
    iteration = 0
    for epoch_num in range(num_epochs):
        epoch_loss = []
        model.train()
        for it in train_dataloader:
            inp, inp_pos, out, out_pos = it
            model = model.to(device)
            inp_pos = inp_pos.to(device)
            out_pos = out_pos.to(device)
            out = out.to(device)
            inp = inp.to(device)
            gnd = out[:, 1:].contiguous().view(-1).long()
            optimizer.zero_grad()

            pred = model(inp.long(), inp_pos, out.long(), out_pos)
            loss = loss_func(pred, gnd)
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
            iteration = iteration + 1
        avg_epoch_loss = sum(epoch_loss) / len(epoch_loss)
        val_loss, val_acc = val(model, val_dataloader, loss_func, batch_size)
        loss_hist = avg_epoch_loss / (batch_size * 4)
        print(
            f"[epoch: {epoch_num+1}]",
            "[loss: ",
            f"{loss_hist:.4f}",
            "]",
            "val_loss: [val_loss ",
            f"{val_loss:.4f}",
            "]",
        )

    return model


def val(model, dataloader, loss_func, batch_size, device=torch.device("cpu")):
    model.eval()
    epoch_loss = []
    num_correct = 0
    total = 0
    for it in dataloader:
        inp, inp_pos, out, out_pos = it

        model = model.to(device)
        inp_pos = inp_pos.to(device)
        out_pos = out_pos.to(device)
        out = out.to(device)
        inp = inp.to(device)
        gnd = out[:, 1:].contiguous().view(-1).long()
        pred = model(inp.long(), inp_pos, out.long(), out_pos)
        loss = loss_func(pred, gnd)

        pred_max = pred.max(1)[1]
        gnd = gnd.contiguous().view(-1)

        n_correct = pred_max.eq(gnd)
        n_correct = n_correct.sum().item()
        num_correct = num_correct + n_correct

        total = total + len(pred_max)
        epoch_loss.append(loss.item())

    avg_epoch_loss = sum(epoch_loss) / len(epoch_loss)
    return avg_epoch_loss / (batch_size * 4), n_correct / total


def inference(model, inp_exp, inp_exp_pos, out_pos_exp, out_seq_len):
    model.eval()
    y_init = torch.LongTensor([14]).unsqueeze(0).cuda().view(1, 1)

    ques_emb = model.emb_layer(inp_exp)
    q_emb_inp = ques_emb + inp_exp_pos
    enc_out = model.encoder(q_emb_inp)
    for i in range(out_seq_len - 1):
        ans_emb = model.emb_layer(y_init)
        a_emb_inp = ans_emb + out_pos_exp[:, : y_init.shape[1], :]
        dec_out = model.decoder(a_emb_inp, enc_out, None)
        _, next_word = torch.max(
            dec_out[0, y_init.shape[1] - 1 : y_init.shape[1]], dim=1
        )

        y_init = torch.cat([y_init, next_word.view(1, 1)], dim=1)
    return y_init, model


def draw(data, x, y, ax):
    seaborn.heatmap(
        data,
        xticklabels=x,
        square=True,
        yticklabels=y,
        vmin=0.0,
        vmax=1.0,
        cbar=False,
        ax=ax,
    )


