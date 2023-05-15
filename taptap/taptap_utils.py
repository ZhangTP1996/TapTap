import typing as tp

import numpy as np
import pandas as pd
import torch

from transformers import AutoTokenizer


def _process_imputation(X_train, df):
    def is_number(s):
        try:
            float(s)
            return True
        except ValueError:
            return False
        except TypeError:
            return False

    num_features = X_train.select_dtypes(include=np.number).columns.to_list()
    cat_features = X_train.select_dtypes(exclude=np.number).columns.to_list()
    for f in num_features:
        df[f] = df[f].apply(lambda x: x if is_number(x) else np.nan)
    for f in cat_features:
        v = X_train[f].astype(str).unique()
        df[f] = df[f].apply(lambda x: x if str(x) in v else np.nan)
    df[num_features] = df[num_features].astype(float)
    df[num_features] = df[num_features].astype(X_train[num_features].dtypes)
    df[cat_features] = df[cat_features].astype(X_train[cat_features].dtypes)
    if num_features:
        df[num_features] = df[num_features].fillna(df[num_features].mean())
    if cat_features:
        df[cat_features] = df[cat_features].fillna(df[cat_features].mode().iloc[0])
    return df


def _array_to_dataframe(data: tp.Union[pd.DataFrame, np.ndarray], columns=None) -> pd.DataFrame:
    """ Converts a Numpy Array to a Pandas DataFrame

    Args:
        data: Pandas DataFrame or Numpy NDArray
        columns: If data is a Numpy Array, columns needs to be a list of all column names

    Returns:
        Pandas DataFrame with the given data
    """
    if isinstance(data, pd.DataFrame):
        return data

    assert isinstance(data, np.ndarray), "Input needs to be a Pandas DataFrame or a Numpy NDArray"
    assert columns, "To convert the data into a Pandas DataFrame, a list of column names has to be given!"
    assert len(columns) == len(data[0]), \
        "%d column names are given, but array has %d columns!" % (len(columns), len(data[0]))

    return pd.DataFrame(data=data, columns=columns)


def _get_column_distribution(df: pd.DataFrame, col: str) -> tp.Union[list, dict]:
    """ Returns the distribution of a given column. If continuous, returns a list of all values.
        If categorical, returns a dictionary in form {"A": 0.6, "B": 0.4}

    Args:
        df: pandas DataFrame
        col: name of the column

    Returns:
        Distribution of the column
    """
    if df[col].dtype == "float":
        col_dist = df[col].to_list()
    else:
        col_dist = df[col].value_counts(1).to_dict()
    return col_dist

def _get_string(numerical_modeling, numerical_features,
                feature, value):
    if numerical_modeling == 'original':
        return "%s is %s" % (feature, value)
    elif numerical_modeling == 'numsplit':
        if feature not in numerical_features or value == 'None':
            return "%s is %s" % (feature, value)
        else:
            s = "%s is" % feature
            if '.' in value:
                tmp = value.split('.')[1]
                tmp = min(3, len(tmp))
                value = f"%.{tmp}f" % float(value)
            for v in value:
                s += ' %s' % v
            return s
    else:
        if feature not in numerical_features:
            return "%s is %s" % (feature, value)
        else:
            s = "%s is" % feature
            c = 0
            if value == 'None':
                s += ' None'
                return s
            value = float(value)
            while value > 1 or value < -1:
                value /= 10
                c += 1
            value = '%.2f' % value
            s += f' {c} @ '
            if value[0] == '-':
                s += '-'
                value = value[1:]
            value = value[2:]
            s += value
            return s


def _convert_tokens_to_text(tokens: tp.List[torch.Tensor], tokenizer: AutoTokenizer) -> tp.List[str]:
    """ Decodes the tokens back to strings

    Args:
        tokens: List of tokens to decode
        tokenizer: Tokenizer used for decoding

    Returns:
        List of decoded strings
    """
    # Convert tokens to text
    text_data = [tokenizer.decode(t) for t in tokens]

    # Clean text
    text_data = [d.replace("<|endoftext|>", "") for d in text_data]
    text_data = [d.replace("\n", " ") for d in text_data]
    text_data = [d.replace("\r", "") for d in text_data]

    return text_data


def _convert_text_to_tabular_data(text: tp.List[str], df_gen: pd.DataFrame, cat_dist=None,
                                  numerical_features=None,
                                  numerical_modeling='original',
                                  prompt_lines=0,
                                  is_print: bool = False) -> pd.DataFrame:
    """ Converts the sentences back to tabular data

    Args:
        text: List of the tabular data in text form
        df_gen: Pandas DataFrame where the tabular data is appended

    Returns:
        Pandas DataFrame with the tabular data from the text appended
    """
    columns = df_gen.columns.to_list()
    columns = [c.strip() for c in columns]
    if cat_dist is None:
        cat_dist = {}
    if numerical_features is None:
        numerical_features = []
        
    # Convert text to tabular data
    df_list = [df_gen]
    for t in text:
        if prompt_lines:
            try:
                t = t.split("Generated example: ")[1]
            except IndexError:
                continue
            except:
                import traceback
                print(traceback.format_exc())
        features = t.split(",")
        td = dict.fromkeys(columns)
        # Transform all features back to tabular data
        for f in features:
            values = f.strip().split(" is ")
            values[0] = values[0].strip()
            # print(values[0], values[0] in columns)
            if values[0] in columns and not td[values[0]]:
                if values[0] in cat_dist and len(values) > 1 and \
                        values[1] not in cat_dist[values[0]]:
                    td[values[0]] = [None]
                    continue
                if numerical_modeling == 'numsplit' and values[0] in numerical_features and len(values) > 1:
                    values[1] = values[1].replace(" ", "")
                try:
                    td[values[0]] = [values[1]]
                except IndexError:
                    #print("An Index Error occurred - if this happends a lot, consider fine-tuning your model further.")
                    pass
        # if not isinstance(td[columns[0]], list):
        #     for key in td: td[key] = [td[key]]
        df_list.append(pd.DataFrame(td))
    df_gen = pd.concat(df_list, ignore_index=True, axis=0)
    return df_gen

