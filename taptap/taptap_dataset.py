import random
import typing as tp
from .taptap_utils import _get_string
import pandas as pd
from datasets import Dataset
from dataclasses import dataclass
from transformers import DataCollatorWithPadding
import numpy as np


class TaptapDataset(Dataset):
    """ GReaT Dataset

    The GReaTDataset overwrites the _getitem function of the HuggingFace Dataset Class to include the permutation step.

    Attributes:
        tokenizer (AutoTokenizer): Tokenizer from HuggingFace
    """

    def set_args(self,
                 numerical_features,
                 target=None,
                 numerical_modeling='original',
                 max_tokens=1024,
                 shuffled_idx=None
                 ):
        self.numerical_features = numerical_features
        self.target = target
        self.numerical_modeling=numerical_modeling
        self.max_tokens = max_tokens
        self.shuffled_idx = shuffled_idx

    def set_tokenizer(self, tokenizer):
        """ Set the Tokenizer

        Args:
            tokenizer: Tokenizer from HuggingFace
        """
        self.tokenizer = tokenizer


    def _getitem(self, key: tp.Union[int, slice, str], decoded: bool = True, **kwargs) -> tp.Union[tp.Dict, tp.List]:
        """ Get Item from Tabular Data

        Get one instance of the tabular data, permuted, converted to text and tokenized.
        """
        # If int, what else?
        shuffled_text = ""
        # for k in [key, np.random.randint(0, len(self._data))]:

        row = self._data.fast_slice(key, 1)
        if self.shuffled_idx is None:
            shuffle_idx = list(range(row.num_columns-1))
            random.shuffle(shuffle_idx)
        else:
            shuffle_idx = self.shuffled_idx

        shuffled_text += ", ".join(
            [_get_string(self.numerical_modeling, self.numerical_features,
                         row.column_names[i], str(row.columns[i].to_pylist()[0]).strip())
             for i in shuffle_idx]
        )
        if random.random() < 0.0001:
            print(shuffled_text)
        tokenized_text = self.tokenizer(shuffled_text)
        tokenized_text['input_ids'] = tokenized_text['input_ids'][:self.max_tokens]
        tokenized_text['attention_mask'] = tokenized_text['attention_mask'][:self.max_tokens]
        return tokenized_text


class MyDataset(Dataset):
    def __init__(self, tokenizer, numerical_modeling,
                 max_tokens):
        self.mydata = []
        self.length = 0
        self.idx = 0
        self.reverse_idx = {}
        self.subtractor = {}
        self.tokenizer = tokenizer
        self.numerical_modeling = numerical_modeling
        self.max_tokens = max_tokens

    def set_tokenizer(self, tokenizer):
        """ Set the Tokenizer

        Args:
            tokenizer: Tokenizer from HuggingFace
        """
        self.tokenizer = tokenizer

    def add_dataframe(self, df: pd.DataFrame):
        self.subtractor[self.idx] = self.length
        for i in range(self.length, self.length+df.shape[0]):
            self.reverse_idx[i] = self.idx
        self.length += df.shape[0]
        self.idx += 1
        numerical_features = df.select_dtypes(include=np.number).columns.to_list()
        great_ds = TaptapDataset.from_pandas(df)
        great_ds.set_args(self.n_line, False, numerical_features=numerical_features,
                          numerical_modeling=self.numerical_modeling,
                          prompt_lines=self.prompt_lines,
                          max_tokens=self.max_tokens)
        great_ds.set_tokenizer(self.tokenizer)
        self.mydata.append(great_ds)


    def __len__(self):
        return self.length

    def _getitem(self, key: tp.Union[int, slice, str], decoded: bool = True, **kwargs) -> tp.Union[tp.Dict, tp.List]:
        """ Get Item from Tabular Data

        Get one instance of the tabular data, permuted, converted to text and tokenized.
        """
        # If int, what else?
        idx = self.reverse_idx[key]
        tokenized_text = self.mydata[idx]._getitem(key-self.subtractor[idx])
        return tokenized_text


@dataclass
class TaptapDataCollator(DataCollatorWithPadding):
    """ GReaT Data Collator

    Overwrites the DataCollatorWithPadding to also pad the labels and not only the input_ids
    """
    def __call__(self, features: tp.List[tp.Dict[str, tp.Any]]):
        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch["labels"] = batch["input_ids"].clone()
        return batch
