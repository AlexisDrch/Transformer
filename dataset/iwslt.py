from typing import Iterable

from torch.utils import data
from torchtext import data, datasets

from dataset.europarl import Split
from dataset.formatter import BatchMasker
from dataset.language_pairs import LanguagePair

ROOT_DATASET_DIR = "resources/torchtext"


class IWSLTDatasetBuilder():
    @staticmethod
    def masked(batch_iterator: Iterable[data.Batch]):
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
              start_token="<s>", eos_token="</s>", blank_token="<blank>", batch_size=32):
        """
        Initializes an iterator over the IWSLT dataset.
        The iterator then yields batches of size `batch_size`.

        Returns the iterator alongside the input & output vocab sets.

        Example:

        >>> dataset_iterator, src_vocab, trg_vocab = IWSLTDatasetBuilder.build(
        ...                                               language_pair=language_pair,
        ...                                               split=Split.Train,
        ...                                               max_length=5,
        ...                                               batch_size=batch_size)
        >>> batch = next(iter(dataset_iterator))

        :param language_pair: The language pair for which to create a vocabulary.
        :param split: The split type.
        :param max_length: Max length of sequence.
        :param min_freq: The minimum frequency a word should have to be included in the vocabulary
        :param start_token: The token that marks the beginning of a sequence.
        :param eos_token: The token that marks an end of sequence.
        :param blank_token: The token to pad with.
        :param batch_size: Desired size of each batch.
        """
        # load corresponding tokenizer
        source_tokenizer, target_tokenizer = language_pair.tokenizer()
        # create pytorchtext data field to generate vocabulary
        source_field = data.Field(tokenize=source_tokenizer, pad_token=blank_token)
        target_field = data.Field(tokenize=target_tokenizer, init_token=start_token,
                                  eos_token=eos_token, pad_token=blank_token)

        # Generates train and validation datasets
        # noinspection PyTypeChecker
        train, validation = datasets.IWSLT.splits(
            root=ROOT_DATASET_DIR,  # To check if the dataset was already downloaded
            exts=language_pair.extensions(),
            fields=(source_field, target_field),
            test=None,
            filter_pred=lambda x: all(len(val) <= max_length for val in (x.src, x.trg))
        )

        if split == Split.Train:
            dataset = train
        elif split == Split.Validation:
            dataset = validation
        else:
            raise NotImplementedError()

        # Build vocabulary on training set
        source_field.build_vocab(train, min_freq=min_freq)
        target_field.build_vocab(train, min_freq=min_freq)

        return IWSLTDatasetBuilder.masked(
            IWSLTDatasetBuilder.transposed(
                data.BucketIterator(
                    dataset=dataset, batch_size=batch_size,
                    sort_key=lambda x: data.interleave_keys(len(x.src), len(x.trg))
                )
            )
        ), source_field.vocab, target_field.vocab
