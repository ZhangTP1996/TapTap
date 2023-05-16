import random
import numpy as np
import typing as tp
import pandas as pd


def _get_sentence(row):
    sentence = ''
    idx_list = row.index.to_list()
    random.shuffle(idx_list)
    for idx in idx_list:
        if pd.isna(row.loc[idx]):
            sentence += '%s is None, ' % (idx)
        else:
            sentence += '%s is %s, ' % (idx, row.loc[idx])
    return sentence

def _get_start_sentence(df, n_lines):
    s = ''
    for i in range(n_lines):
        s += 'Prompt example: '
        s += _get_sentence(df.iloc[i])
    s = s[:-2]
    s += '. Generated example: '
    return s


class TaptapStart:
    """ Abstract super class GReaT Start

    GReaT Start creates tokens to start the generation process.

    Attributes:
        tokenizer (AutoTokenizer): Tokenizer, automatically downloaded from llm-checkpoint
    """
    def __init__(self, tokenizer):
        """
        Initializes the super class.

        Args:
            tokenizer: Tokenizer from the HuggingFace library
        """
        self.tokenizer = tokenizer

    def get_start_tokens(self, n_samples: int, imbalance: bool, numerical_modeling='original') -> tp.List[tp.List[int]]:
        """ Get Start Tokens

        Creates starting points for the generation process

        Args:
            n_samples: Number of start prompts to create

        Returns:
            List of n_sample lists with tokens
        """
        raise NotImplementedError("This has to be overwritten but the subclasses")


class CategoricalStart(TaptapStart):
    """ Categorical Starting Feature

    A categorical column with its categories is used as starting point.

    Attributes:
        start_col (str): Name of the categorical column
        population (list[str]): Possible values the column can take
        weights (list[float]): Probabilities for the individual categories

    """
    def __init__(self, tokenizer, start_col: str, start_col_dist: dict):
        """ Initializes the Categorical Start

        Args:
            tokenizer: Tokenizer from the HuggingFace library
            start_col: Name of the categorical column
            start_col_dist: Distribution of the categorical column (dict of form {"Cat A": 0.8, "Cat B": 0.2})
        """
        super().__init__(tokenizer)

        assert isinstance(start_col, str), ""
        assert isinstance(start_col_dist, dict), ""

        self.start_col = start_col
        self.population = list(start_col_dist.keys())
        self.weights = list(start_col_dist.values())

    def get_start_tokens(self, n_samples, imbalance, numerical_modeling='original'):
        if imbalance:
            values = zip(self.population, self.weights)
            values = sorted(values, key=lambda x: x[1])
            value, _ = values[0]
            start_words = [value] * n_samples
        else:
            start_words = random.choices(self.population, self.weights, k=n_samples)
        start_text = [self.start_col + " is " + str(s) + "," for s in start_words]
        start = self.tokenizer(start_text, return_tensors="pt", padding=True)
        return start

    def get_prompt_lines_start_tokens(self, data, n_samples, n_lines):
        start_words = random.choices(self.population, self.weights, k=n_samples)
        start_sentence = []
        for i in range(n_samples):
            df = data.sample(n=n_lines)
            start_sentence.append(_get_start_sentence(df, n_lines))
        start_text = [start_sentence[i] + self.start_col + " is " + str(s) + "," for i, s in enumerate(start_words)]
        start = self.tokenizer(start_text, return_tensors="pt", padding=True)
        return start


class ContinuousStart(TaptapStart):
    """ Continuous Starting Feature

    A continuous column with some noise is used as starting point.

    Attributes:
        start_col (str): Name of the continuous column
        start_col_dist (list[float]): The continuous column from the train data set
        noise (float): Size of noise that is added to each value
        decimal_places (int): Number of decimal places the continuous values have
    """
    def __init__(self, tokenizer, start_col: str, start_col_dist: tp.List[float],
                 noise: float = .01, decimal_places: int = 5):
        """ Initializes the Continuous Start

        Args:
            tokenizer: Tokenizer from the HuggingFace library
            start_col: Name of the continuous column
            start_col_dist: The continuous column from the train data set
            noise: Size of noise that is added to each value
            decimal_places: Number of decimal places the continuous values have
        """
        super().__init__(tokenizer)

        assert isinstance(start_col, str), ""
        assert isinstance(start_col_dist, list), ""

        self.start_col = start_col
        self.start_col_dist = start_col_dist
        self.noise = noise
        self.decimal_places = decimal_places

    def get_numeracy(self, value):
        s = ''
        value = str(value)
        if '.' in value:
            tmp = value.split('.')[1]
            tmp = min(3, len(tmp))
            value = f"%.{tmp}f" % float(value)
        for v in value:
            s += ' %s' % v
        return s

    def get_start_tokens(self, n_samples, imbalance, numerical_modeling='original'):
        start_words = random.choices(self.start_col_dist, k=n_samples)
        # start_words += np.random.normal(size=n_samples) * self.noise  # add noise to start words
        if numerical_modeling == 'split':
            start_text = [self.start_col + " is" + self.get_numeracy(s) + "," for s in start_words]
        else:
            start_text = [self.start_col + " is " + format(s, f".{self.decimal_places}f") + "," for s in start_words]
        start = self.tokenizer(start_text, return_tensors="pt", padding=True)
        return start

    def get_prompt_lines_start_tokens(self, data, n_samples, n_lines):
        start_words = random.choices(self.start_col_dist, k=n_samples)
        start_sentence = []
        for i in range(n_samples):
            df = data.sample(n=n_lines)
            start_sentence.append(_get_start_sentence(df, n_lines))
        start_text = [start_sentence[i] + self.start_col + " is " + format(s, f".{self.decimal_places}f") + ","
                      for i, s in enumerate(start_words)]
        start = self.tokenizer(start_text, return_tensors="pt", padding=True)
        return start


class RandomStart(TaptapStart):
    """ Random Starting Features

    Random column names are used as start point. Can be used if no distribution of any column is known.

    Attributes:
        all_columns (List[str]): Names of all columns
    """
    def __init__(self, tokenizer, all_columns: tp.List[str]):
        """ Initializes the Random Start

        Args:
            tokenizer: Tokenizer from the HuggingFace library
            all_columns: Names of all columns
        """
        super().__init__(tokenizer)
        self.all_columns = all_columns

    def get_start_tokens(self, n_samples):
        start_words = random.choices(self.all_columns, k=n_samples)
        start_text = [s + " is " for s in start_words]
        start = self.tokenizer(start_text, return_tensors="pt", padding=True)
        return start

    def get_prompt_lines_start_tokens(self, data, n_samples, n_lines):
        start_words = random.choices(self.all_columns, k=n_samples)
        start_sentence = []
        for i in range(n_samples):
            df = data.sample(n=n_lines)
            start_sentence.append(_get_start_sentence(df, n_lines))
        start_text = [start_sentence[i] + s + " is " for i, s in enumerate(start_words)]
        start = self.tokenizer(start_text, return_tensors="pt", padding=True)
        return start


