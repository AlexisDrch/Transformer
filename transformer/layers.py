import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from attention import MultiHeadAttention


class PositionwiseFeedForward(nn.Module):
    """
    2-layers Feed-Forward Network with a ReLU activation & dropout in between.

    The equation is:

    .. math::

        FFN(x) = Dropout(max(0, x \\cdot W1 + b1) ) \\cdot W2 + b2


    """
    "Implements FFN equation."
    def __init__(self, d_model: int, d_ff: int, dropout=0.1):
        """
        Constructor for the 2-layers FFN.

        :param d_model: Dimensionality used across the model. Should be 512.

        :param d_ff: Hidden layer size (should be 2048).

        :param dropout: dropout probability. Default is 0.1

        """
        # call base constructor
        super(PositionwiseFeedForward, self).__init__()

        # first layer
        self.linear_1 = nn.Linear(d_model, d_ff)

        # dropout layer
        self.dropout = nn.Dropout(dropout)

        # second layer
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the FFN.

        :param x: input Tensor, should be 3-dimensional: (batch_size, seq_length, d_model).

        :return: output Tensor, same shape as input.

        """
        return self.linear_2(self.dropout(F.relu(self.linear_1(x))))


class LayerNormalization(nn.Module):
    """
    Implements Layer Normalization. See https://arxiv.org/abs/1607.06450 for reference.

    The equation should be:

    .. math::

        h = \\frac{a}{\\sigma} \\cdot (x − \\mu ) + b

    Where a, b are learnable parameters,  \sigma & \mu are the standard deviation & mean of x.
    """
    def __init__(self, size: int, eps=1e-6):
        """
        Constructor for the ``LayerNormalization`` class.

        :param size: Size of the tensor which will be normalized.
        :param eps: small epsilon value (to avoid ``ZeroDivisionError`` errors)
        """
        super(LayerNormalization, self).__init__()

        # instantiate the learnable parameters.
        self.a_2 = nn.Parameter(torch.ones(size))
        self.b_2 = nn.Parameter(torch.zeros(size))

        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the normalization layer.

        :param x: input tensor to be normalized.

        :return: normalized x.
        """
        mean, std = x.mean(-1, keepdim=True), x.std(-1, keepdim=True)

        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class ResidualConnection(nn.Module):
    """
    A residual connection followed by a layer normalization.

    The overall equation should be:

    .. math::

        LayerNorm(x + Sublayer(x))

    Where ``Sublayer()`` is the function implemented by the sublayer (e.g. ``MultiHeadAttention``).

    Dropout is applied to the output of the sublayer before it is added to the sub-layer input and normalized.

    """
    def __init__(self, size: int, dropout: float):
        """
        Constructor for the ``ResidualConnection`` class.

        :param size: Size of the input tensor.

        :param dropout: Dropout probability.
        """
        # call base constructor
        super(ResidualConnection, self).__init__()

        # layer normalization
        self.norm = LayerNormalization(size)

        # dropout layer
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, sublayer: nn.Module) -> Tensor:
        """
        Apply the residual connection to any ``sublayer`` with the same size.

        :param x: Input tensor, which will be fed to the sublayer, then summed with the residual and normalized.
        :param sublayer: Sublayer class (e.g. ``MultiHeadAttention``).

        :return: Normalized tensor after the residual connection.
        """
        ""
        return self.norm(x + self.dropout(sublayer(x)))


if __name__ == '__main__':
    batch_size = 64
    sequence_length = 10
    dim = 512

    x = torch.ones((batch_size, sequence_length, dim))

    ffn = PositionwiseFeedForward(d_model=dim, d_ff=2048, dropout=0.1)

    ffn_output = ffn(x)

    multi_attention = MultiHeadAttention(n_head=8, d_model=512, d_k=64, d_v=64)

    residual_connection = ResidualConnection(size=512, dropout=0.1)

    res_output = residual_connection(x=x, sublayer=lambda x: multi_attention(x, x, x))
    print(res_output.shape)
