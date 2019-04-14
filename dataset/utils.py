from enum import IntEnum, auto

import nltk


class Tokenizer(object):
    """
    Tokenizes the text, i.e. segments it into words, punctuation and so on.
    This is done by applying rules specific to each language.
    For example, punctuation at the end of a sentence should be split off
        – whereas “U.K.” should remain one token.
    """

    def __init__(self, language: str):
        """
        Loads the appropriate model from NLTK

        :param language: model string id
        """
        nltk.download(info_or_id="punkt", quiet=True)
        self.language = language

    def __call__(self, text: str):
        """
        tokenize a string in corresponding token
        :param text: String to be tokenized.
        :return: List of tokens.
        """
        return nltk.word_tokenize(text, language=self.language)


class Split(IntEnum):
    Train = auto()
    Validation = auto()
