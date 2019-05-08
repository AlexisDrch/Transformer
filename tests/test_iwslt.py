from os import getenv
from unittest import TestCase, skipIf

from dataset.iwslt import IWSLTDatasetBuilder
from dataset.language_pairs import LanguagePair
from dataset.utils import Split


class TestIWSLT(TestCase):
    @skipIf(len(getenv("CI", "")) > 0, "skipping slow tests on CI")
    def test_build_train_batches(self):
        batch_size = 64
        language_pair = LanguagePair.fr_en

        dataset_iterator, val_iterator, test_iterator, _, _ = (
            IWSLTDatasetBuilder.build(language_pair=language_pair,
                                      split=Split.Train,
                                      max_length=100, batch_size_train=batch_size)
        )
        self.assertIsNotNone(dataset_iterator)
        self.assertIsNone(val_iterator)
        self.assertIsNone(test_iterator)
        batch = next(iter(dataset_iterator))

    @skipIf(len(getenv("CI", "")) > 0, "skipping slow tests on CI")
    def test_build_test_batches(self):
        batch_size = 64
        language_pair = LanguagePair.fr_en

        dataset_iterator, val_iterator, test_iterator, _, _ = (
            IWSLTDatasetBuilder.build(language_pair=language_pair,
                                      split=Split.Test,
                                      max_length=100, batch_size_train=batch_size)
        )
        self.assertIsNone(dataset_iterator)
        self.assertIsNone(val_iterator)
        self.assertIsNotNone(test_iterator)

    @skipIf(len(getenv("CI", "")) > 0, "skipping slow tests on CI")
    def test_build_validation_batches(self):
        batch_size = 96
        language_pair = LanguagePair.fr_en

        dataset_iterator, val_iterator, test_iterator, _, _ = (
            IWSLTDatasetBuilder.build(language_pair=language_pair,
                                      split=Split.Validation,
                                      max_length=40, batch_size_train=batch_size)
        )
        self.assertIsNone(dataset_iterator)
        self.assertIsNone(test_iterator)
        self.assertIsNotNone(val_iterator)
        print(len(list(val_iterator)))
        pass