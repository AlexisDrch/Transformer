from unittest import TestCase

import torch

from transformer.attention import MultiHeadAttention
from transformer.encoder import EncoderLayer, Encoder
from transformer.layers import PositionwiseFeedForward


class TestEncoder(TestCase):
    def test_forward(self):
        enc_layer = EncoderLayer(
            size=512,
            self_attention=MultiHeadAttention(n_head=8, d_model=512,
                                              d_k=64, d_v=64, dropout=0.1),
            feed_forward=PositionwiseFeedForward(d_model=512, d_ff=2048, dropout=0.1),
            dropout=0.1
        )

        encoder = Encoder(layer=enc_layer, n_layers=6)
        x = torch.ones((64, 10, 512))

        out = encoder.forward(x, mask=None, verbose=False)

        self.assertIsInstance(out, torch.Tensor)
        self.assertEqual(out.shape, x.shape)
        # check no nan values
        self.assertEqual(torch.isnan(out).sum(), 0)
