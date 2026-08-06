import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTM(nn.Module):
    """Single-layer, uni-directional LSTM module."""

    def __init__(self, input_dim: int, hidden_dim: int):
        """
        Initialize a LSTM. Model parameters to initialize:
            Wx: Weights for input-to-hidden connections, of shape (D, 4H)
            Wh: Weights for hidden-to-hidden connections, of shape (H, 4H)
            b: Biases, of shape (4H,)

        Args:
            input_dim: Input size, denoted as D before
            hidden_dim: Hidden size, denoted as H before
        """
        super().__init__()

        # Register parameters
        self.Wx = nn.Parameter(
            torch.randn(input_dim, hidden_dim * 4).div(math.sqrt(input_dim))
        )
        self.Wh = nn.Parameter(
            torch.randn(hidden_dim, hidden_dim * 4).div(math.sqrt(hidden_dim))
        )
        self.b = nn.Parameter(torch.zeros(hidden_dim * 4))

    def step_forward(
        self, x: torch.Tensor, prev_h: torch.Tensor, prev_c: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for a single timestep of an LSTM.
        The input data has dimension D, the hidden state has dimension H, and
        we use a minibatch size of N.

        Args:
            x: Input data for one time step, of shape (N, D)
            prev_h: The previous hidden state, of shape (N, H)
            prev_c: The previous cell state, of shape (N, H)
            Wx: Input-to-hidden weights, of shape (D, 4H)
            Wh: Hidden-to-hidden weights, of shape (H, 4H)
            b: Biases, of shape (4H,)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]
                next_h: Next hidden state, of shape (N, H)
                next_c: Next cell state, of shape (N, H)
        """
        ######################################################################
        # TODO: Implement the forward pass for a single timestep of an LSTM.
        ######################################################################
        next_h, next_c = None, None
        # out = torch.cat([x, prev_h], dim=1) @ torch.cat([self.Wx, self.Wh], dim=0) + self.b
        out = x @ self.Wx + prev_h @ self.Wh + self.b # N, 4H
        i, f, o, g = torch.chunk(out, chunks=4, dim=1)
        i = F.sigmoid(i) # N, H
        f = F.sigmoid(f) # N, H
        o = F.sigmoid(o) # N, H
        g = F.tanh(g) # N, H
        next_c = f * prev_c + i * g
        next_h = o * F.tanh(next_c)
        
        
        return next_h, next_c

    def forward(self, x: torch.Tensor, h0: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for an LSTM over an entire sequence of data. We assume an
        input sequence composed of T vectors, each of dimension D. The LSTM
        uses a hidden size of H, and we work over a minibatch containing N
        sequences. After running the LSTM forward, we return the hidden states
        for all timesteps.

        Note that the initial cell state is passed as input, but the initial
        cell state is set to zero. Also note that the cell state is not returned;
        it is an internal variable to the LSTM and is not accessed from outside.

        Args:
            x: Input data for the entire timeseries, of shape (N, T, D)
            h0: Initial hidden state, of shape (N, H)

        Returns:
            hn: The hidden state output.
        """

        c0 = torch.zeros_like(h0)  # we provide the intial cell state c0 here for you!
        ######################################################################
        # TODO: Implement the forward pass for an LSTM over entire timeseries
        ######################################################################
        hn = None
        
        N, T, _ = x.shape
        prev_h = h0
        prev_c = c0
        hn = []
        for t in range(T):
            next_h, next_c = self.step_forward(x[:,t,:], prev_h, prev_c)
            
            hn.append(next_h)
            prev_c = next_c
            prev_h = next_h
        hn = torch.stack(hn, dim=1)
        

        return hn


