from unittest import TestCase

import numpy as np
import torch

from transformer.embeddings import Embeddings, PositionalEncoding


class TestEmbeddings(TestCase):
    def test_forward(self):
        out_size = np.random.randint(10, 100)
        vocab_size = np.random.randint(100, 1000)
        nb_words = np.random.randint(10, 100)

        # Generate input and one-hot-encode it
        input = torch.randint(0, vocab_size, (nb_words,))
        one_hots = torch.eye(vocab_size, dtype=torch.long)[input, :]

        # Create model
        embeddings = Embeddings(d_model=out_size, vocab_size=vocab_size)

        # Run forward pass
        out = embeddings.forward(one_hots)

        # Check output
        self.assertIsInstance(out, torch.Tensor)
        self.assertEqual(out.shape, (nb_words, vocab_size, out_size))


class TestPositionalEncoding(TestCase):
    def test_forward(self):
        # Positional Encoding:
        batch_size = 64
        sequence_length = 10
        d_model = 512

        pos_encoding = PositionalEncoding(d_model, 0.1)

        # verify that the positional encoding do not require gradients
        self.assertEqual(pos_encoding.pos_encoding.requires_grad, False)

        x = torch.ones((batch_size, sequence_length, d_model), requires_grad=True)

        output = pos_encoding(x)

        # check type
        self.assertIsInstance(output, torch.Tensor)

        # check shape
        self.assertEqual(output.shape, x.shape)

        # verify that the output of the forward pass still requires gradients
        self.assertEqual(output.requires_grad, True)
        # check no nan values
        self.assertEqual(torch.isnan(x).sum(), 0)
