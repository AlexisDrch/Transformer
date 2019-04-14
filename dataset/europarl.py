import numpy as np
from torch.utils import data

from dataset.language_pairs import LanguagePair
from dataset.utils import Split


class Europarl(data.Dataset):
    def __init__(self, language: LanguagePair, split: Split, split_size=0.6):
        self.language = language
        self.split = split
        # Load file in memory
        path_lang1, path_lang2 = language.europarl_path()
        with open(path_lang1) as f1, open(path_lang2) as f2:
            self.array = [tuple(l.strip() for l in lines) for lines in zip(f1, f2)]

        # Train/Val split
        n = len(self.array)
        self.indexes = np.arange(n)
        np.random.shuffle(self.indexes)

        train_size = round(n * split_size)
        if split == Split.Train:
            self.indexes = self.indexes[:train_size]
        elif split == Split.Validation:
            self.indexes = self.indexes[train_size:]
        else:
            raise NotImplementedError()

        # load corresponding tokenizers
        self.source_tokenizer, self.target_tokenizer = language.tokenizer()

    def __getitem__(self, index):
        source, target = self.array[self.indexes[index]]
        return source, target

    def __len__(self):
        return len(self.indexes)
