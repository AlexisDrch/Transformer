import torch
from unittest import TestCase
from transformer.model import Transformer


class TestEncoder(TestCase):
    def test_forward(self):
        """
        Test the model forward pass (i.e. not training it) on a batch of randomly created samples.

        """
        params = {
            'd_model': 512,
            'src_vocab_size': 27000,
            'tgt_vocab_size': 27000,

            'N': 6,
            'dropout': 0.1,

            'attention': {'n_head': 8,
                          'd_k': 64,
                          'd_v': 64,
                          'dropout': 0.1},

            'feed-forward': {'d_ff': 2048,
                             'dropout': 0.1},
        }

        # 1. test constructor
        transformer = Transformer(params)

        # 2. test forward pass
        batch_size = 64
        input_sequence_length = 10
        output_sequence_length = 13

        # create a batch of random samples
        src = torch.randint(low=1, high=params["src_vocab_size"], size=(batch_size, input_sequence_length))
        trg = torch.randint(low=1, high=params["tgt_vocab_size"], size=(batch_size, output_sequence_length))

        # create masks for src & trg: assume pad_token=0. Since we draw randomly in [1, upper_bound), the mask is only 1s.
        src_mask = torch.ones_like(src).unsqueeze(-2)
        trg_mask = torch.ones_like(trg).unsqueeze(-2)

        logits = transformer(src_sequences=src, src_mask=src_mask, trg_sequences=trg, trg_mask=trg_mask)

        self.assertIsInstance(logits, torch.Tensor)
        self.assertEqual(logits.shape, torch.Size([batch_size, output_sequence_length, params['tgt_vocab_size']]))
        # check no nan values
        self.assertEqual(torch.isnan(logits).sum(), 0)
