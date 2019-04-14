from unittest import TestCase

from dataset.europarl import Europarl, Split
from dataset.language_pairs import LanguagePair


class TestEuroparl(TestCase):
    def test_getitem(self):
        batch_size = 64
        dataset = Europarl(language=LanguagePair.fr_en, split=Split.Train, split_size=0.6)
        source, target = dataset[0]

        # unit tests
        self.assertIsInstance(source, str)
        self.assertIsInstance(target, str)
