import torch
import torch.nn as nn
from torch import Tensor

from utils import clones, BColors, subsequent_mask
from layers import PositionwiseFeedForward, LayerNormalization, ResidualConnection
from attention import ScaledDotProductAttention, MultiHeadAttention


class Decoder(nn.Module):
    """
    Implementation of the Decoder of the Transformer model.

    Constituted of a stack of N identical layers.
    """

    def __init__(self, layer: nn.Module, N: int):
        """
        Constructor for the global Decode
        :param layer: layer module to use.
        :param N: Number of decoder layers to use.
        """

        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNormalization(layer.size)

    def forward(self, x: Tensor, memory: Tensor, self_mask: Tensor, memory_mask: None, verbose=False) -> Tensor:
        """
        Forward pass: Relays the output of layer i to layer i+1
        :param x: input Tensor of decoder
        :param memory: output from encoder
        :param self_mask: Mask to be used in the self-attention sub-module.
                Prevent predictions from attending to subsequent positions: i.e Prediction at position i can depend only
                on the known outputs at positions less than i.
        :param memory_mask: Mask to be used in the memory-attention sub-module. Optional.
        :param verbose: Whether to add debug/info messages or not.
        :return:
        """

        for i, layer in enumerate(self.layers):
            if verbose:
                print(f'Going into layer {i}')
            x = layer(x, memory, self_mask, memory_mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    """
        Implements one Decoder layer. The actual Decoder is a stack of N of these layers.

        The overall forward pass of this layer is as follows:

        ------------------------------> memory (Encoder output)
                                          |
                                          v
        x -> self-attn -> add & norm -> memory-attn -> add & norm -> feed_forward -> add & norm -> output
            |              ^           |               ^            |                ^
            v -----------> |           v ------------> |            v -------------> |
    """

    def __init__(self, size: int, self_attn: nn.Module, memory_attn: nn.Module, feed_forward: nn.Module, dropout: float):
        """
        Constructor for the ``DecoderLayer`` class.
        :param size: Input size
        :param self_attn: Class used for the self-attention part of the layer.
        :param memory_attn: Class used for the memory-attention part of the layer.
        :param feed_forward: Class used for the feed-forward part of the layer.
        :param dropout: dropout probability
        """

        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.memory_attn = memory_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(ResidualConnection(size, dropout), 3)

    def forward(self, x: Tensor, memory: Tensor, self_mask: Tensor, memory_mask=None) -> Tensor:
        """
        :param x: Input Tensor, should be 3-dimensional: (batch_size, seq_length, d_model).
                Should represent the input sentences or the output of the previous Encoder layer.
        :param memory: Memory input Tensor, should be 3-dimensional: (batch_size, seq_length, d_model).
                Should represent the memory output of the Encoder layer.
        :param self_mask: Mask to be used in the self-attention sub-module.
                Prevent predictions from attending to subsequent positions: i.e Prediction at position i can depend only
                on the known outputs at positions less than i.
        :param memory_mask: Mask to be used in the memory-attention sub-module. Optional.
        :return: Output of the DecoderLayer, should be of the same shape as the input.
        """

        # multi-head attention over the input of the decoder
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, self_mask))
        # multi-head attention over the output of the encoder stack
        x = self.sublayer[1](x, lambda x: self.memory_attn(x, memory, memory, memory_mask))
        # final feed forward with residual & norm
        return self.sublayer[2](x, self.feed_forward)


if __name__ == '__main__':

    # initialization parameters
    batch_size = 64
    sequence_length = 10
    d_k = d_v = d_model = input_size = 512
    d_ff = 2048
    N_decoder_layer = 6

    # initialization decoder
    decoder_layer = DecoderLayer(size=input_size,
                                 self_attn=MultiHeadAttention(n_head=8, d_model=d_model, d_k=d_k, d_v=d_v, dropout=0.1),
                                 memory_attn=MultiHeadAttention(n_head=8, d_model=d_model, d_k=d_k, d_v=d_v, dropout=0.1),
                                 feed_forward=PositionwiseFeedForward(d_model=d_model, d_ff=d_ff, dropout=0.1),
                                 dropout=0.1)

    decoder = Decoder(layer=decoder_layer, N=N_decoder_layer)

    # initialization input and memory
    x = torch.ones((batch_size, sequence_length, input_size))
    memory = torch.ones((batch_size, sequence_length, input_size))

    # subsequent mask: mask all words > i
    decoder_mask = subsequent_mask(sequence_length)

    # forward pass with fool input and memory (same here)
    out = decoder(x, memory, decoder_mask, None)

    # forward unit test
    assert out.shape == x.shape
    assert isinstance(out, Tensor)
    assert out.shape == memory.shape
    assert x.shape == memory.shape

    print(out.shape)
    print(f'{BColors.OKGREEN}-Decoder forward() unit tests: passed')




