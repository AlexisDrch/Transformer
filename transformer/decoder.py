import torch.nn as nn
from torch import Tensor

from transformer.layers import LayerNormalization, ResidualConnection
from transformer.utils import clone


class Decoder(nn.Module):
    """
    Implementation of the Decoder of the Transformer model.

    Constituted of a stack of ``N`` identical layers.
    """

    def __init__(self, layer: nn.Module, N: int):
        """
        Constructor for the global ``Decoder``.

        :param layer: layer module to use.

        :param N: number of decoder layers to use.
        """
        # call base constructor
        super(Decoder, self).__init__()

        self.layers = clone(layer, N)

        self.norm = LayerNormalization(layer.size)

    def forward(self, x: Tensor, memory: Tensor, self_mask: Tensor, memory_mask: Tensor,
                verbose=False) -> Tensor:
        """
        Forward pass: Relays the output of layer `i` to layer `i+1`.

        :param x: input Tensor of decoder. Should be of shape (batch_size, seq_len, d_model).

        :param memory: Output of the ``Encoder`` stack. Should be of same shape as ``x``.

        :param self_mask: Mask hiding both the padding in ``x`` and prevent predictions from attending to subsequent
        positions.

        .. note::

            This mask prevents predictions from attending to subsequent positions:

            i.e Prediction at position i can depend only on the known outputs at positions less than i.

            This mask also hides padding, i.e. mark the sequence's length padding element so that they can be ignored.


        :param memory_mask: Corresponding mask of ``memory``.

        :param verbose: Whether to add debug/info messages or not.
        """

        for i, layer in enumerate(self.layers):
            if verbose:
                print(f"Going into layer {i}")
            x = layer(x, memory, self_mask, memory_mask)

        return self.norm(x)


class DecoderLayer(nn.Module):
    """
        Implements one ``Decoder`` layer. The actual :py:class:`Decoder` is a stack of ``N`` of these layers.

        The overall forward pass of this layer is as follows:

        ------------------------------> memory (``Encoder`` output)
                                          |
                                          v
        x -> self-attn -> add & norm -> memory-attn -> add & norm -> feed_forward -> add & norm -> output
            |              ^           |               ^            |                ^
            v -----------> |           v ------------> |            v -------------> |
    """

    def __init__(self, size: int, self_attn: nn.Module, memory_attn: nn.Module,
                 feed_forward: nn.Module, dropout: float):
        """
        Constructor for the ``DecoderLayer`` class.

        :param size: Input size.

        :param self_attn: Class used for the self-attention part of the layer (e.g. ``MultiHeadAttention`` with mask).

        :param memory_attn: Class used for the memory-attention part of the layer (e.g. ``MultiHeadAttention``).

        :param feed_forward: Class used for the feed-forward part of the layer (e.g ``PositionwiseFeedForward``).

        :param dropout: dropout probability.
        """

        super(DecoderLayer, self).__init__()
        self.size = size

        # self attention sub-module
        self.self_attn = self_attn

        # memory-attention sub-module: link with Encoder
        self.memory_attn = memory_attn

        # feed-forward sub-module
        self.feed_forward = feed_forward

        # residual connections
        self.sublayer = clone(ResidualConnection(size, dropout), 3)

    def forward(self, x: Tensor, memory: Tensor, self_mask: Tensor, memory_mask: Tensor) -> Tensor:
        """
        :param x: Input Tensor, should be 3-dimensional: (batch_size, seq_length, d_model).
                Should represent the output of the previous Decoder layer or teacher-forcing inputs (or start token).

        :param memory: Output of the ``Encoder`` stack. Should be of same shape as ``x``.

        :param self_mask: Mask hiding both the padding in ``x`` and prevent predictions from attending to subsequent
        positions.

        .. note::

            This mask prevents predictions from attending to subsequent positions:

            i.e Prediction at position i can depend only on the known outputs at positions less than i.

            This mask also hides padding, i.e. mark the sequence's length padding element so that they can be ignored.


        :param memory_mask: Corresponding mask of ``memory``.


        :return: Output of the ``DecoderLayer``, should be of the same shape as the input.
        """

        # multi-head attention over the input of the decoder
        out_self_attn = self.sublayer[0](x, lambda y: self.self_attn(y, y, y, self_mask))

        # multi-head attention over the output of the encoder stack
        out_memory_attn = self.sublayer[1](out_self_attn, lambda y: self.memory_attn(y, memory, memory, memory_mask))

        # final feed forward with residual & norm
        return self.sublayer[2](out_memory_attn, self.feed_forward)
