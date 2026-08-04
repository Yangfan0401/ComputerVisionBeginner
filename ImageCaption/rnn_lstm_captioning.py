import math
import pprint
from turtle import ht
from typing import Optional, Tuple  # noqa: UP035

import torch
import torchvision
from sympy import sequence
from torch import mode, ne, nn
from torch.nn import functional as F
from sklearn import feature_extraction

import sys
from pathlib import Path

from ImageCaption.a5_helper import ImageEncoder, WordEmbedding, dot_product_attention
from ImageCaption.models.atten_lstm import AttentionLSTM
from ImageCaption.models.lstm import LSTM
from ImageCaption.models.rnn import RNN
sys.path.append(str(Path(__file__).parent.parent))

def hello_rnn_lstm_captioning():
    print("Hello from rnn_lstm_captioning.py!")



class CaptioningRNN(nn.Module):
    """
    A CaptioningRNN produces captions from images using a recurrent
    neural network.

    The RNN receives input vectors of size D, has a vocab size of V, works on
    sequences of length T, has an RNN hidden dimension of H, uses word vectors
    of dimension W, and operates on minibatches of size N.

    Note that we don't use any regularization for the CaptioningRNN.

    You will implement the `__init__` method for model initialization and
    the `forward` method first, then come back for the `sample` method later.
    """

    def __init__(
            self,
            word_to_idx,
            input_dim: int = 512,
            wordvec_dim: int = 128,
            hidden_dim: int = 128,
            cell_type: str = "rnn",
            image_encoder_pretrained: bool = True,
            ignore_index: Optional[int] = None,
    ):
        """
        Construct a new CaptioningRNN instance.

        Args:
            word_to_idx: A dictionary giving the vocabulary. It contains V
                entries, and maps each string to a unique integer in the
                range [0, V).
            input_dim: Dimension D of input image feature vectors.
            wordvec_dim: Dimension W of word vectors.
            hidden_dim: Dimension H for the hidden state of the RNN.
            cell_type: What type of RNN to use; either 'rnn' or 'lstm'.
        """
        super().__init__()
        if cell_type not in {"rnn", "lstm", "attn"}:
            raise ValueError('Invalid cell_type "%s"' % cell_type)

        self.cell_type = cell_type
        self.word_to_idx = word_to_idx
        self.idx_to_word = {i: w for w, i in word_to_idx.items()}

        vocab_size = len(word_to_idx)

        self._null = word_to_idx["<NULL>"]
        self._start = word_to_idx.get("<START>", None)
        self._end = word_to_idx.get("<END>", None)
        self.ignore_index = ignore_index

        ######################################################################
        # TODO: Initialize the image captioning module. Refer to the TODO
        # in the captioning_forward function on layers you need to create
        #
        # You may want to check the following pre-defined classes:
        # ImageEncoder WordEmbedding, RNN, LSTM, AttentionLSTM, nn.Linear
        #
        # (1) output projection (from RNN hidden state to vocab probability)
        # (2) feature projection (from CNN pooled feature to h0)
        ######################################################################
        # Replace "pass" statement with your code
        self.image_encoder = ImageEncoder(image_encoder_pretrained) #RNN features extraction
        self.feat_proj = nn.Linear(input_dim, hidden_dim) #Fully-Connected
        # torch.nn.init.normal(self.feat_proj.weight, mean=0, std=0.01)
        # torch.nn.init.zeros_(self.feat_proj.bias)
        self.attn_proj = nn.Linear(input_dim, hidden_dim)
        # torch.nn.init.normal(self.attn_proj.weight, mean=0, std=0.01)
        # torch.nn.init.zeros_(self.attn_proj.bias)

        self.word_embedding = WordEmbedding(vocab_size, wordvec_dim)

        if cell_type == "rnn":
            self.rnn = RNN(wordvec_dim, hidden_dim)
        elif cell_type == "lstm":
            self.lstm = LSTM(wordvec_dim, hidden_dim)
        elif cell_type == "attn":
            self.attn_lstm = AttentionLSTM(wordvec_dim, hidden_dim)

        self.score_logits = nn.Linear(hidden_dim, vocab_size)
        # torch.nn.init.normal(self.score_logits.weight, mean=0, std=0.01)
        # torch.nn.init.zeros_(self.score_logits.bias)
        ######################################################################
        #                            END OF YOUR CODE                        #
        ######################################################################

    def forward(self, images, captions):
        """
        Compute training-time loss for the RNN. We input images and the GT
        captions for those images, and use an RNN (or LSTM) to compute loss. The
        backward part will be done by torch.autograd.

        Args:
            images: Input images, of shape (N, 3, 112, 112)
            captions: Ground-truth captions; an integer array of shape (N, T + 1)
                where each element is in the range 0 <= y[i, t] < V

        Returns:
            loss: A scalar loss
        """
        # Cut captions into two pieces: captions_in has everything but the last
        # word and will be input to the RNN; captions_out has everything but the
        # first word and this is what we will expect the RNN to generate. These
        # are offset by one relative to each other because the RNN should produce
        # word (t+1) after receiving word t. The first element of captions_in
        # will be the START token, and the first element of captions_out will
        # be the first word.
        captions_in = captions[:, :-1]
        captions_out = captions[:, 1:]

        loss = 0.0
        ######################################################################
        # TODO: Implement the forward pass for the CaptioningRNN.
        # In the forward pass you will need to do the following:
        # (1) Use an affine transformation to project the image feature to
        #     the initial hidden state $h0$ (for RNN/LSTM, of shape (N, H)) or
        #     the projected CNN activation input $A$ (for Attention LSTM,
        #     of shape (N, H, 4, 4).
        # (2) Use a word embedding layer to transform the words in captions_in
        #     from indices to vectors, giving an array of shape (N, T, W).
        # (3) Use either a vanilla RNN or LSTM (depending on self.cell_type) to
        #     process the sequence of input word vectors and produce hidden state
        #     vectors for all timesteps, producing an array of shape (N, T, H).
        # (4) Use a (temporal) affine transformation to compute scores over the
        #     vocabulary at every timestep using the hidden states, giving an
        #     array of shape (N, T, V).
        # (5) Use (temporal) softmax to compute loss using captions_out, ignoring
        #     the points where the output word is <NULL>.
        #
        # Do not worry about regularizing the weights or their gradients!
        ######################################################################
        # Replace "pass" statement with your code
        sequence = None
        features = self.image_encoder(images)
        N, D, _, _ = features.shape
        pooling_features = features.mean(dim=(2, 3))
        h0 = self.feat_proj(pooling_features)
        word_vectors = self.word_embedding(captions_in).to(dtype=h0.dtype, device=h0.device)
        if self.cell_type == "rnn":
            sequence = self.rnn(word_vectors, h0)
        elif self.cell_type == "lstm":
            sequence = self.lstm(word_vectors, h0)
        elif self.cell_type == "attn":
            H, D = self.attn_proj.weight.shape
            attn_features = features.reshape(N, D, -1).permute(0, 2, 1)
            A = self.attn_proj(attn_features).permute(0, 2, 1).reshape(shape=[N, H, 4, 4])
            sequence = self.attn_lstm(word_vectors, A)

        scores = self.score_logits(sequence)
        loss = temporal_softmax_loss(scores, captions_out, self.ignore_index)
        ######################################################################
        #                           END OF YOUR CODE                         #
        ######################################################################

        return loss

    def sample(self, images, max_length=15):
        """
        Run a test-time forward pass for the model, sampling captions for input
        feature vectors.

        At each timestep, we embed the current word, pass it and the previous hidden
        state to the RNN to get the next hidden state, use the hidden state to get
        scores for all vocab words, and choose the word with the highest score as
        the next word. The initial hidden state is computed by applying an affine
        transform to the image features, and the initial word is the <START>
        token.

        For LSTMs you will also have to keep track of the cell state; in that case
        the initial cell state should be zero.

        Args:
            images: Input images, of shape (N, 3, 112, 112)
            max_length: Maximum length T of generated captions

        Returns:
            captions: Array of shape (N, max_length) giving sampled captions,
                where each element is an integer in the range [0, V). The first
                element of captions should be the first sampled word, not the
                <START> token.
        """
        N = images.shape[0]
        captions = self._null * images.new(N, max_length).fill_(1).long()

        if self.cell_type == "attn":
            attn_weights_all = images.new(N, max_length, 4, 4).fill_(0).float()

        ######################################################################
        # TODO: Implement test-time sampling for the model. You will need to
        # initialize the hidden state of the RNN by applying the learned affine
        # transform to the image features. The first word that you feed to
        # the RNN should be the <START> token; its value is stored in the
        # variable self._start. At each timestep you will need to do to:
        # (1) Embed the previous word using the learned word embeddings
        # (2) Make an RNN step using the previous hidden state and the embedded
        #     current word to get the next hidden state.
        # (3) Apply the learned affine transformation to the next hidden state to
        #     get scores for all words in the vocabulary
        # (4) Select the word with the highest score as the next word, writing it
        #     (the word index) to the appropriate slot in the captions variable
        #
        # For simplicity, you do not need to stop generating after an <END> token
        # is sampled, but you can if you want to.
        #
        # NOTE: we are still working over minibatches in this function. Also if
        # you are using an LSTM, initialize the first cell state to zeros.
        # For AttentionLSTM, first project the 1280x4x4 CNN feature activation
        # to $A$ of shape Hx4x4. The LSTM initial hidden state and cell state
        # would both be A.mean(dim=(2, 3)).
        #######################################################################
        # Replace "pass" statement with your code
        prev_h, prev_cell,A = None, None, None
        T = 15
        features = self.image_encoder(images)
        N, H, _, _ = features.shape
        pooling_features = features.mean(dim=(2, 3))
        prev_h = self.feat_proj(pooling_features)
        prev_word = torch.full((N, T), self._start)
        if self.cell_type == "lstm":
            prev_cell = torch.zeros_like(prev_h)
        elif self.cell_type == "attn":
            H, D = self.attn_proj.weight.shape
            attn_features = features.reshape(N, D, -1).permute(0, 2, 1)
            A = self.attn_proj(attn_features).permute(0, 2, 1).reshape(shape=[N, H, 4, 4])
            prev_h = A.mean(dim=(2, 3))
            prev_cell = prev_h

        for t in range(max_length):

            word_vectors = self.word_embedding(prev_word).to(dtype=prev_h.dtype, device=prev_h.device)

            if self.cell_type == "lstm":
                next_h, next_c = self.lstm.step_forward(
                    word_vectors[:, t, :], prev_h, prev_cell
                )
                prev_cell = next_c
            elif self.cell_type == "rnn":
                next_h = self.rnn.step_forward(word_vectors[:, t, :], prev_h)
            elif self.cell_type == "attn":
                attn, attn_weights = dot_product_attention(prev_h, A)
                next_h, next_c = self.attn_lstm.step_forward(word_vectors[:, t,:], prev_h, prev_cell, attn)
                prev_cell = next_c

            prev_h = next_h

            scores = self.score_logits(next_h)
            highest_scores = scores.argmax(dim=1)
            captions[:, t] = highest_scores
            prev_word[:, t] = highest_scores

        ######################################################################
        #                           END OF YOUR CODE                         #
        ######################################################################
        if self.cell_type == "attn":
            return captions, attn_weights_all.cpu()
        else:
            return captions

