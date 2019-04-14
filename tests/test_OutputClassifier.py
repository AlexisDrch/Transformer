from unittest import TestCase

import torch
from transformer.classifier import OutputClassifier


class TestOutputClassifier(TestCase):
    def test_forward(self):
        batch_size = 64
        sequence_length = 10
        d_model = 512

        vocab = 37000  # source-target vocabulary size

        x = torch.ones((batch_size, sequence_length, d_model))

        out_classifier = OutputClassifier(d_model=d_model, vocab=vocab)

        output = out_classifier(x)

        # check type
        self.assertIsInstance(output, torch.Tensor)

        # check tensor shape
        self.assertEqual(output.shape, torch.Size([batch_size, sequence_length, vocab]))

        # check no nan values
        self.assertEqual(torch.isnan(output).sum(), 0)
