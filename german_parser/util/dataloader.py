import xml.etree.ElementTree as ET
import random
from math import floor
from string import punctuation
from torch.utils.data import Dataset
from collections.abc import Callable
from typing import Literal
import torch

from .logger import model_logger
from .c_and_d import ConstituentTree


class TigerDataset(Dataset):
    def __init__(self, in_sentences: list[list[str]], use_new_words: bool, word_dict: dict[str, int], character_dict: dict[str, int], character_flag_generators: list[Callable[[str], Literal[1, 0]]]) -> None:
        super().__init__()

        self.word_dict = word_dict
        self.character_dict = character_dict
        self.character_flag_generators = character_flag_generators
        self.use_new_words = use_new_words

        self.num_sentences = len(in_sentences)
        self.max_sentence_length = max(len(s) for s in in_sentences)

        # define dictionaries for new words. These dictionaries will be shared across batches to save generating them every time a batch is created
        self.new_words_to_id: dict[str, int] = {}
        for sent in in_sentences:
            for word in sent:
                if word not in self.word_dict and word not in self.new_words_to_id:
                    self.new_words_to_id[word] = len(self.new_words_to_id) + 1
        self.id_to_new_words: dict[int, str] = {i: s for s, i in self.new_words_to_id.items()}

        self.sentence_lengths = torch.zeros(self.num_sentences, dtype=torch.long)
        self.data = torch.ones(self.num_sentences, self.max_sentence_length, dtype=torch.long) # 1 corresponds to <PAD>

        for i, sent in enumerate(in_sentences):
            self.sentence_lengths[i] = len(sent)
            for j, word in enumerate(sent):
                if word in self.word_dict:
                    self.data[i, j] = self.word_dict[word]
                elif self.use_new_words:
                    self.data[i, j] = -self.new_words_to_id[word]
                else:
                    self.data[i, j] = 0 # 0 corresponds to <UNK>


    def __len__(self):
        return self.num_sentences
    
    def __getitem__(self, idx):
        """get item

        Args:
            idx (indices): indices

        Returns:
            tuple[torch.Tensor, torch.Tensor, dict[int, str] | None]: data, sentence lengths, new words dictionary (if self.use_new_words is True)
        """
        return self.data[idx], self.sentence_lengths[idx], (self.id_to_new_words if self.use_new_words else None)

class TigerDatasetGenerator:
    def _set_word_dict(self, coverage: float): 
        # CREATE WORD AND CHARACTER DICTIONARIES
        all_word_freqs: dict[str, int] = {} # words by word count
        num_all_words = 0
        for sent in self.sentences:
            for word in sent.get_words():
                num_all_words += 1
                if word not in all_word_freqs:
                    all_word_freqs[word] = 1
                all_word_freqs[word] += 1

        all_words_sorted = sorted(all_word_freqs.keys(), key=lambda k: all_word_freqs[k], reverse=True)
        w_idx = 0
        cum_freq = 0
        while cum_freq < num_all_words * coverage:
            cum_freq += all_word_freqs[all_words_sorted[w_idx]]
            w_idx += 1

        self.word_dict: dict[str, int] = {w: i + 1 for i, w in enumerate(all_words_sorted[:w_idx])}

    def _set_character_dict(self):
        all_characters_lower: set[str] = set()
        all_characters_upper: set[str] = set()

        for sent in self.sentences:
            for word in sent.get_words():
                all_characters_lower.update(word.lower())
                all_characters_upper.update(word.upper())

        self.character_set: dict[str, int] = {c: i + 2 for i, c in enumerate(all_characters_lower)}
        for u in all_characters_upper:
            if u not in self.character_set:
                self.character_set[u] = self.character_set[u.lower()]

    def _set_character_flag_generators(self):
        self.character_flag_generators: list[Callable[[str], Literal[1, 0]]] = [ # type: ignore
            lambda c: int(c.isupper()),
            lambda c: int(c.lower() in ["ä", "ö", "ü", "ß"]),
            lambda c: int(c.isdigit()),
            lambda c: int(c in punctuation)
        ]

    def __init__(self, file_path: str, split: tuple[float, float], vocab_coverage: float=0.95, prop_of_tiger_to_use: float=1.0):
        """initializes dataset generator

        Args:
            file_path (str): path to tiger.xml
            split (tuple[float, float]): (dev_proportion, test_proportion)
            vocab_coverage (float): used to determine vocabulary size: vocabulary will be large enough to cover (vocab_coverage) of the words in the training sentence
            prop_of_tiger_to_use (float): proportion of tiger dataset to use (defaults to 1.0)
        """

        # PARSE XML
        model_logger.info(f"Parsing dataset from '{file_path}'...")        
        with open(file_path, "rb") as f:
            self.document = ET.parse(f)

        all_sentences = self.document.findall(".//s")
        all_sentences = random.sample(all_sentences, floor(len(all_sentences) * prop_of_tiger_to_use))
        num_all_sentences = len(all_sentences)

        model_logger.info(f"Parsed {num_all_sentences} sentences.")

        # CONVERT INTO CONSTITUENCY TREES
        model_logger.info(f"Generating trees...")

        errors = 0
        has_unary = 0
        has_empty_verb_constituents = 0

        cannot_use = 0

        error_messages = []

        self.sentences: list[ConstituentTree] = []
        for idx, s in enumerate(all_sentences):
            s_cannot_use = False

            try:
                sent = ConstituentTree.from_tiger_xml(s)
                has_unary += int(sent.has_unary)
                has_empty_verb_constituents += int(sent.has_empty_verb_constituents)

                s_cannot_use = sent.has_unary or sent.has_empty_verb_constituents

                if not s_cannot_use:
                    self.sentences.append(sent)
            except Exception as e:
                errors += 1
                error_messages.append(f"sentence id {idx + 1} has error {e}")

                s_cannot_use = True

            cannot_use += int(s_cannot_use)
        
        self.num_sentences = len(self.sentences)
        model_logger.info(f"{self.num_sentences} ({100 * self.num_sentences / num_all_sentences:.2f}%) trees generated.")

        # SPLIT DATASET
        dev_start_idx = floor(self.num_sentences * (1 - split[0] - split[1]))
        test_start_idx = floor(self.num_sentences * (1 - split[1]))

        self.train_sentences = self.sentences[:dev_start_idx]
        self.dev_sentences = self.sentences[dev_start_idx:test_start_idx]
        self.test_sentences = self.sentences[test_start_idx:]

        model_logger.info(f"Dataset split into {len(self.train_sentences)} training, {len(self.dev_sentences)} dev, and {len(self.test_sentences)} test trees.")

        self._set_word_dict(vocab_coverage)
        self._set_character_dict()
        self._set_character_flag_generators()

    def _get_dataset(self, dataset: list[ConstituentTree], use_new_words: bool=True):
        return TigerDataset([s.get_words() for s in dataset], use_new_words, self.word_dict, self.character_set, self.character_flag_generators)

    def get_training_dataset(self, use_new_words: bool=True):
        return self._get_dataset(self.train_sentences, use_new_words)
    
    def get_dev_dataset(self, use_new_words: bool=True):
        return self._get_dataset(self.dev_sentences, use_new_words)
    
    def get_test_dataset(self, use_new_words: bool=True):
        return self._get_dataset(self.test_sentences, use_new_words)

# TODO: perturb dataset by removing capitals and umlauts, and using ae instead of ä