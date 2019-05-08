from typing import Iterable

from torch.utils import data
from torchtext import data, datasets

from dataset.utils import Split
from dataset.formatter import BatchMasker
from dataset.language_pairs import LanguagePair

ROOT_DATASET_DIR = "resources/torchtext"


class IWSLTDatasetBuilder():
    @staticmethod
    def masked(batch_iterator: Iterable[data.Batch]):
        """
        Helper generator to mask a batch
        """
        for batch in batch_iterator:
            yield BatchMasker(batch)

    @staticmethod
    def transposed(batch_iterator: Iterable[data.Batch]):
        """
        Transposes each batch in a batch iterator.
        This is needed as BucketIterator generates iterator with
        dimensions (n, batch_size), when we want (batch_size, n).

        :param batch_iterator: A batch iterator whose batches we want to transpose.
        """
        for batch in batch_iterator:
            batch.src.transpose_(0, 1)
            batch.trg.transpose_(0, 1)
            yield batch

    @staticmethod
    def build(language_pair: LanguagePair, split: Split, max_length=100, min_freq=2,
              start_token="<s>", eos_token="</s>", blank_token="<blank>",
              batch_size_train=32, batch_size_validation=32,
              batch_size_test=32, device='cpu'):
        """
        Initializes an iterator over the IWSLT dataset.
        The iterator then yields batches of size `batch_size`.

        Returns one iterator for each split alongside the input & output vocab sets.

        Example:

        >>> dataset_iterator, _, _, src_vocab, trg_vocab = IWSLTDatasetBuilder.build(
        ...                                                   language_pair=language_pair,
        ...                                                   split=Split.Train,
        ...                                                   max_length=5,
        ...                                                   batch_size_train=batch_size_train)
        >>> batch = next(iter(dataset_iterator))

        :param language_pair: The language pair for which to create a vocabulary.
        :param split: The split type.
        :param max_length: Max length of sequence.
        :param min_freq: The minimum frequency a word should have to be included in the vocabulary
        :param start_token: The token that marks the beginning of a sequence.
        :param eos_token: The token that marks an end of sequence.
        :param blank_token: The token to pad with.
        :param batch_size_train: Desired size of each training batch.
        :param batch_size_validation: Desired size of each validation batch.
        :param batch_size_test: Desired size of each testing batch.
        :param device: The device on which to store the batches.
        :type device: str or torch.device

        :returns: (train_iterator, validation_iterator, test_iterator,
                   source_field.vocab, target_field.vocab)
        """
        # load corresponding tokenizer
        source_tokenizer, target_tokenizer = language_pair.tokenizer()
        # create pytorchtext data field to generate vocabulary
        source_field = data.Field(tokenize=source_tokenizer, pad_token=blank_token)
        target_field = data.Field(tokenize=target_tokenizer, init_token=start_token,
                                  eos_token=eos_token, pad_token=blank_token)

        # Generates train and validation datasets
        settings = dict()
        for key, split_type in [
            # ("validation", Split.Validation),  # Due to a bug in TorchText, cannot set to None
            ("test", Split.Test),
        ]:
            if (split & split_type):
                pass  # Keep default split setting
            else:
                settings[key] = None  # Disable split
        # noinspection PyTypeChecker
        train, validation, *out = datasets.IWSLT.splits(
            root=ROOT_DATASET_DIR,  # To check if the dataset was already downloaded
            exts=language_pair.extensions(),
            fields=(source_field, target_field),
            filter_pred=lambda x: all(len(val) <= max_length for val in (x.src, x.trg)),
            **settings
        )

        # Build vocabulary on training set
        source_field.build_vocab(train, min_freq=min_freq)
        target_field.build_vocab(train, min_freq=min_freq)

        train_iterator, validation_iterator, test_iterator = None, None, None

        def sort_func(x):
            return data.interleave_keys(len(x.src), len(x.trg))

        if split & Split.Train:
            train_iterator = data.BucketIterator(
                dataset=train, batch_size=batch_size_train, repeat=False,
                device=device, sort_key=sort_func)
        if split & Split.Validation:
            validation_iterator = data.BucketIterator(
                dataset=validation, batch_size=batch_size_validation, repeat=False,
                device=device, sort_key=sort_func)
        if split & Split.Test:
            test, *out = out
            test_iterator = data.BucketIterator(
                dataset=test, batch_size=batch_size_test, repeat=False,
                device=device, sort_key=sort_func)

        return (
            train_iterator,
            validation_iterator,
            test_iterator,
            source_field.vocab,
            target_field.vocab,
        )
