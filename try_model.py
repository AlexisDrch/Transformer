import argparse

import torch

from dataset.iwslt import IWSLTDatasetBuilder
from dataset.language_pairs import LanguagePair
from dataset.utils import Split
from training.loss import LabelSmoothingLoss
from transformer.model import Transformer


def get_args():
    parser = argparse.ArgumentParser(description='PyTorch Model Tester')
    parser.add_argument('--model-path',
                        type=str,
                        help='Path to the model to test.')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    batch_size = 1024
    smoothing = 0.
    print("Loading dataset...")
    _, val_iterator, _, src_vocab, trg_vocab = (
        IWSLTDatasetBuilder.build(language_pair=LanguagePair.fr_en,
                                  split=Split.Validation,
                                  max_length=40, batch_size_train=batch_size)
    )
    print(f"Loading model from '{args.model_path}'...")
    model = Transformer.load_model_from_file(args.model_path, params={
        'd_model': 512,
        'N': 6,
        'dropout': 0.1,
        'src_vocab_size': len(src_vocab),
        'tgt_vocab_size': len(trg_vocab),

        'attention': {
            'n_head': 8,
            'd_k': 64,
            'd_v': 64,
            'dropout': 0.1},

        'feed-forward': {
            'd_ff': 2048,
            'dropout': 0.1
        }
    })

    print("Computing loss on validation set...")
    loss_fn = LabelSmoothingLoss(size=len(trg_vocab),
                                 padding_token=src_vocab.stoi['<blank>'],
                                 smoothing=smoothing)
    val_loss = 0.
    with torch.no_grad():
        for i, batch in enumerate(
            IWSLTDatasetBuilder.masked(
                IWSLTDatasetBuilder.transposed(
                    val_iterator
                ))):

            # Convert batch to CUDA.
            if torch.cuda.is_available():
                batch.cuda()

            # 1. Perform forward pass.
            logits = model(batch.src, batch.src_mask, batch.trg, batch.trg_mask)

            # 2. Evaluate loss function.
            loss = loss_fn(logits, batch.trg_shifted)

            # Accumulate loss
            val_loss += loss.item()

    print("Done.")
    print(f"Batch size: {batch_size} | Smoothing = {smoothing}")
    print(f"Validation Loss: {val_loss} / {(i+1)} = {val_loss/(i+1)}")
