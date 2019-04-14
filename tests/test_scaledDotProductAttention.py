from unittest import TestCase

import torch
from transformer.attention import ScaledDotProductAttention

class TestScaledDotProductAttention(TestCase):
    def test_forward(self):
        batch_size = 64
        sequence_length = 10
        d_k = d_v = 512

        queries = torch.ones((batch_size, sequence_length, d_k))
        keys = torch.ones((batch_size, sequence_length, d_k))

        values = torch.ones((batch_size, sequence_length, d_v))

        scaled_attention = ScaledDotProductAttention()

        output = scaled_attention(queries=queries, keys=keys, values=values)

        self.assertIsInstance(output, torch.Tensor)
        self.assertEqual(output.shape, values.shape)
        # check no nan values
        self.assertEqual(torch.isnan(output).sum(), 0)
