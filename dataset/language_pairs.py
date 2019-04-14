from enum import IntEnum, auto

from dataset.utils import Tokenizer


class LanguagePair(IntEnum):
    """
    An enumeration of all language pair configurations available.
    """
    fr_en = auto()

    def tokenizer(self):
        if self == LanguagePair.fr_en:
            return (
                Tokenizer(language='french'),
                Tokenizer(language='english'),
            )
        else:
            raise ValueError()

    def extensions(self):
        if self == LanguagePair.fr_en:
            return ('.fr', '.en')
        else:
            raise ValueError()

    def europarl_path(self):
        if self == LanguagePair.fr_en:
            return (
                "resources/europarl/fr-en/europarl-v7.fr-en.fr.big.txt",
                "resources/europarl/fr-en/europarl-v7.fr-en.en.big.txt"
            )
