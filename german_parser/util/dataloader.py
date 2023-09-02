import xml.etree.ElementTree as ET
import random
from math import floor
from string import punctuation
from torch.utils.data import Dataset
from collections.abc import Callable
from typing import Literal
import torch
from torch import nn
from torch.utils.data import DataLoader, default_collate

from .logger import model_logger
from .c_and_d import ConstituentTree, DependencyTree
from .const import CONSTS


class TigerDataset(Dataset):
    def __init__(self, in_dependency_trees: list[DependencyTree], use_new_words: bool, word_dict: dict[str, int], character_dict: dict[str, int],
                 sym_dict: dict[str, int], character_flag_generators: list[Callable[[str], Literal[1, 0]]]) -> None:
        super().__init__()

        self.word_dict = word_dict
        self.character_dict = character_dict
        self.sym_dict = sym_dict
        self.character_flag_generators = character_flag_generators
        self.use_new_words = use_new_words

        self.num_sentences = len(in_dependency_trees)
        self.max_sentence_length = max(len(s.get_words()) for s in in_dependency_trees)

        # define dictionaries for new words. These dictionaries will be shared across batches to save generating them every time a batch is created
        self.new_words_to_id: dict[str, int] = {}
        for d_tree in in_dependency_trees:
            for word in d_tree.get_words():
                if word not in self.word_dict and word not in self.new_words_to_id:
                    self.new_words_to_id[word] = len(self.new_words_to_id) + 1
        self.id_to_new_words: dict[int, str] = {i: s for s, i in self.new_words_to_id.items()}

        self.sentence_lengths = torch.zeros(self.num_sentences, dtype=torch.long)
        self.data = torch.ones(self.num_sentences, self.max_sentence_length, dtype=torch.long) # ones to default with padding
        self.heads = torch.zeros(self.num_sentences, self.max_sentence_length, dtype=torch.long)
        self.syms = torch.zeros(self.num_sentences, self.max_sentence_length, dtype=torch.long)
        self.attachment_orders = torch.zeros(self.num_sentences, self.max_sentence_length, dtype=torch.long) # use max_sentence_length for redundancy; TODO: change later. 1-indexed

        self.dependency_trees = in_dependency_trees

        for i, d_tree in enumerate(in_dependency_trees):
            self.sentence_lengths[i] = d_tree.num_words

            for j, word in enumerate(d_tree.get_words()):
                if word in self.word_dict:
                    self.data[i, j] = self.word_dict[word]
                elif self.use_new_words:
                    self.data[i, j] = -self.new_words_to_id[word]
                else:
                    self.data[i, j] = 0 # 0 corresponds to <UNK>

            syms_in_tree = d_tree.get_syms()

            for head_idx, modifier_idxs in d_tree.get_tree_map().items():
                attachment_orders = d_tree.modifiers[head_idx].keys() # these will not be 1-indexed, so we need to translate between internal attachment orders and 1-indexed attachment orders
                attachment_order_dict: dict[int, int] = {val: zero_indexed + 1 for zero_indexed, val in enumerate(sorted(attachment_orders))} # internal-keys to 1-indexed keys
                for attachment_order in attachment_orders:
                    for child in d_tree.modifiers[head_idx][attachment_order]:
                        modifier_idx = child.modifier
                        self.attachment_orders[i, modifier_idx - 1] = attachment_order_dict[attachment_order]

                for modifier_idx in modifier_idxs:
                    # since the dependency tree is 1-indexed, we need to subtract 1 from the indices. HOWEVER: we add 1 back onto head indices, since head index 0 corresponds to no head (root)
                    self.heads[i, modifier_idx - 1] = head_idx
                    self.syms[i, modifier_idx - 1] = self.sym_dict[syms_in_tree[modifier_idx]]

            # note: root word has been neglected in syms and attachment order, so deal with them now:
            root_idx = d_tree.get_root()
            self.syms[i, root_idx - 1] = self.sym_dict[CONSTS["d_tree_root_sym"]]
            self.attachment_orders[i, root_idx - 1] = 1

    @classmethod
    def _sorted_collate(cls, word_codes: torch.Tensor, sentence_lengths: torch.Tensor, head_targets: torch.Tensor, sym_targets: torch.Tensor, attachment_order_targets: torch.Tensor):
        """given a batch of word codes, sentence lengths, and head targets, sorts them by sentence length in descending order, and returns the sorted batch, truncated to remove unnecessary padding 1s

        Args:
            word_codes (torch.Tensor): (B, T), where T is the maximum sentence length across the entire dataset
            sentence_lengths (torch.Tensor): (B)
            head_targets (torch.Tensor): (B, T)

        Returns:
            _type_: tuple[word_codes, sentence_lengths, head_targets] which have been sorted and such that their sizes are:
                word_codes (B, T_batch)
                sentence_lengths (B)
                head_targets (B, T_batch),
            where T_batch is maximum sentence length within the batch. This removes unnecessary padidng 1s at the end of each row of word_codes and head_targets
        """
        arg_sort = sentence_lengths.argsort(descending=True)

        sentence_lengths_sorted = sentence_lengths[arg_sort]
        T_batch = sentence_lengths_sorted[0] # maximum sentence size within the batch
        
        word_codes_sorted: torch.Tensor = word_codes[arg_sort][:, :T_batch]
        head_targets_sorted: torch.Tensor = head_targets[arg_sort][:, :T_batch]
        sym_targets_sorted: torch.Tensor = sym_targets[arg_sort][:, :T_batch]
        attachment_order_targets_sorted: torch.Tensor = attachment_order_targets[arg_sort][:, :T_batch]

        return (word_codes_sorted, sentence_lengths_sorted, head_targets_sorted, sym_targets_sorted, attachment_order_targets_sorted)

    def __len__(self):
        return self.num_sentences
    
    def __getitem__(self, idx):
        """get item

        Args:
            idx (indices): indices

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: data, sentence lengths, head targets
        """

        if idx == int(idx):
            return self.data[idx], self.sentence_lengths[idx], self.heads[idx], self.syms[idx], self.attachment_orders[idx]
        else:
            idx = torch.as_tensor(idx)
            return self._sorted_collate(self.data[idx], self.sentence_lengths[idx], self.heads[idx], self.syms[idx], self.attachment_orders[idx])

    def get_new_words_dict(self):
        return self.id_to_new_words

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

        self.word_dict: dict[str, int] = {w: i + 2 for i, w in enumerate(all_words_sorted[:w_idx])} # 0 and 1 are reserved for <UNK> and <PAD> respectively
        self.inverse_word_dict: dict[int, str] = {i: w for w, i in self.word_dict.items()}

    def _set_character_dict(self):
        all_characters_lower: set[str] = set()
        all_characters_upper: set[str] = set()

        for sent in self.sentences:
            for word in sent.get_words():
                all_characters_lower.update(word.lower())
                all_characters_upper.update(word.upper())

        self.character_set: dict[str, int] = {c: i + 2 for i, c in enumerate(all_characters_lower)} # 0 and 1 are reserved for <UNK> and <PAD> respectively
        for u in all_characters_upper:
            if u not in self.character_set:
                self.character_set[u] = self.character_set[u.lower()]


    def _set_sym_dict(self):
        """generate sym dictionary. keys are 0-indexed, where 0 corresponds to DROOT
        """
        all_syms: set[str] = set().union(*[c.get_all_syms() for c in self.sentences])
        self.sym_set: dict[str, int] = {sym: i + 1 for i, sym in enumerate(all_syms)}
        self.sym_set[CONSTS["d_tree_root_sym"]] = 0
        self.inverse_sym_set: dict[int, str] = {i: sym for sym, i in self.sym_set.items()}

    def __init__(self, file_path: str, split: tuple[float, float], vocab_coverage: float=0.95, prop_of_tiger_to_use: float=1.0, character_flag_generators: list[Callable[[str], Literal[0, 1]]] = []):
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

                s_cannot_use = sent.has_unary or sent.has_empty_verb_constituents or sent.get_num_words() < 2

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
        self._set_sym_dict()

        self.character_flag_generators = character_flag_generators

    def _get_dataset(self, dataset: list[ConstituentTree], use_new_words: bool=True) -> TigerDataset:
        return TigerDataset(
            [DependencyTree.from_c_tree(s) for s in dataset],
            use_new_words,
            self.word_dict,
            self.character_set,
            self.sym_set,
            self.character_flag_generators
        )

    def _get_dataloader(self, dataset: TigerDataset, **kargs) -> tuple[DataLoader[TigerDataset], dict[int, str]]:
        return DataLoader(
            dataset,
            **kargs,
            collate_fn=lambda batch : TigerDataset._sorted_collate(*default_collate(batch))
        ), dataset.get_new_words_dict()

    def get_training_dataset(self, use_new_words: bool=True) -> TigerDataset:
        return self._get_dataset(self.train_sentences, use_new_words)
    
    def get_dev_dataset(self, use_new_words: bool=True) -> TigerDataset:
        return self._get_dataset(self.dev_sentences, use_new_words)
    
    def get_test_dataset(self, use_new_words: bool=True) -> TigerDataset:
        return self._get_dataset(self.test_sentences, use_new_words)
    
    def get_training_dataloader(self, use_new_words: bool=True, **kargs) -> tuple[DataLoader[TigerDataset], dict[int, str]]:
        return self._get_dataloader(self.get_training_dataset(use_new_words), **kargs)
    
    def get_dev_dataloader(self, use_new_words: bool=True, **kargs) -> tuple[DataLoader[TigerDataset], dict[int, str]]:
        return self._get_dataloader(self.get_dev_dataset(use_new_words), **kargs)
    
    def get_test_dataloader(self, use_new_words: bool=True, **kargs) -> tuple[DataLoader[TigerDataset], dict[int, str]]:
        return self._get_dataloader(self.get_test_dataset(use_new_words), **kargs)

# TODO: perturb dataset by removing capitals and umlauts, and using ae instead of ä