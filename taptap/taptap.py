import os
import warnings
import json
import typing as tp
import logging

import numpy as np
import pandas as pd
import lightgbm as lgb
from tqdm import tqdm
import gc

import torch
# from transformers import (AutoTokenizer, LlamaTokenizer, LlamaForCausalLM,
#                           AutoModelForCausalLM, GPT2LMHeadModel, AutoModel,
#                           TrainingArguments)
from transformers import (AutoTokenizer,
                          AutoModelForCausalLM, GPT2LMHeadModel, AutoModel,
                          TrainingArguments)
from taptap_dataset import TaptapDataset, TaptapDataCollator, MyDataset
from taptap_start import TaptapStart, CategoricalStart, ContinuousStart, RandomStart
from taptap_trainer import TaptapTrainer
from taptap_utils import _array_to_dataframe, _get_column_distribution, _convert_tokens_to_text, \
    _convert_text_to_tabular_data, _get_string, _process_imputation


class Taptap:
    """

    The Taptap class handles the whole generation flow. It is used to fine-tune a large language model for tabular data,
    and to sample synthetic tabular data.

    Attributes:
        llm (str): HuggingFace checkpoint of a pretrained large language model, used a basis of our model
        tokenizer (AutoTokenizer): Tokenizer, automatically downloaded from llm-checkpoint
        model (AutoModelForCausalLM): Large language model, automatically downloaded from llm-checkpoint
        experiment_dir (str): Directory, where the training checkpoints will be saved
        epochs (int): Number of epochs to fine-tune the model
        batch_size (int): Batch size used for fine-tuning
        train_hyperparameters (dict): Additional hyperparameters added to the TrainingArguments used by the
         HuggingFaceLibrary, see here the full list of all possible values
         https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.TrainingArguments
        columns (list): List of all features/columns of the tabular dataset
        num_cols (list): List of all numerical features/columns of the tabular dataset
        conditional_col (str): Name of a feature/column on which the sampling can be conditioned
        conditional_col_dist (dict | list): Distribution of the feature/column specified by condtional_col
    """

    def __init__(self,
                 llm: str,
                 experiment_dir: str = "trainer_taptap",
                 steps: int = 20000,
                 batch_size: int = 8,
                 numerical_modeling: str = "split",
                 max_tokens: int = 1024,
                 **train_kwargs):
        """

        Args:
            llm: HuggingFace checkpoint of a pretrained large language model, used a basis of our model
            experiment_dir:  Directory, where the training checkpoints will be saved
            epochs: Number of epochs to fine-tune the model
            batch_size: Batch size used for fine-tuning
            train_kwargs: Additional hyperparameters added to the TrainingArguments used by the HuggingFaceLibrary,
             see here the full list of all possible values
             https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.TrainingArguments
        """
        # Load Model and Tokenizer from HuggingFace
        self.llm = llm
        self.tokenizer = AutoTokenizer.from_pretrained(self.llm)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(self.llm)

        # Set the training hyperparameters
        self.experiment_dir = experiment_dir
        self.steps = steps
        self.batch_size = batch_size
        self.numerical_modeling = numerical_modeling
        self.max_tokens = max_tokens
        self.train_hyperparameters = train_kwargs

        # Needed for the sampling process
        self.columns = None
        self.num_cols = None
        self.conditional_col = None
        self.conditional_col_dist = None
        self.X_train_impute = None


    def pretrain(self, data_list, numerical_modeling='original',
                 learning_rate=5e-5, resume_from_checkpoint: tp.Union[bool, str] = False):
        logging.info("Convert data into HuggingFace dataset object...")

        my_ds = MyDataset(tokenizer=self.tokenizer,
                          numerical_modeling=numerical_modeling,
                          max_tokens=self.max_tokens
                          )
        for df in data_list:
            my_ds.add_dataframe(df)

        my_ds.set_tokenizer(self.tokenizer)

        # Set training hyperparameters
        logging.info("Create Trainer...")
        training_args = TrainingArguments(self.experiment_dir,
                                          max_steps=self.steps,
                                          learning_rate=learning_rate,
                                          save_steps=self.steps,
                                          ddp_find_unused_parameters=False,
                                          per_device_train_batch_size=self.batch_size,
                                          **self.train_hyperparameters)
        great_trainer = TaptapTrainer(self.model, training_args, train_dataset=my_ds, tokenizer=self.tokenizer,
                                      data_collator=TaptapDataCollator(self.tokenizer))

        # Start training
        logging.info("Start training...")
        great_trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        return great_trainer


    def fit(self, data: tp.Union[pd.DataFrame, np.ndarray], target_col: str, task: str,
            column_names: tp.Optional[tp.List[str]] = None,
            conditional_col: tp.Optional[str] = None, resume_from_checkpoint: tp.Union[bool, str] = False) \
            -> TaptapTrainer:
        """ Fine-tune using tabular data.

        Args:
            data: Pandas DataFrame or Numpy Array that contains the tabular data
            target_col: The target column.
            column_names: If data is Numpy Array, the feature names have to be defined. If data is Pandas
            DataFrame, the value is ignored
            conditional_col: If given, the distribution of this column is saved and used as a starting
            point for the generation process later. If None, the last column is considered as conditional feature
            resume_from_checkpoint: If True, resumes training from the latest checkpoint in the experiment_dir.
            If path, resumes the training from the given checkpoint (has to be a valid HuggingFace checkpoint!)

        Returns:
            TaptapTrainer used for the fine-tuning process
        """
        self.X_train_impute = data.copy()
        numerical_features = data.select_dtypes(include=np.number).columns.to_list()
        df = _array_to_dataframe(data, columns=column_names)
        self._update_column_information(df)
        self._update_conditional_information(df, conditional_col, task)

        # Convert DataFrame into HuggingFace dataset object
        logging.info("Convert data into HuggingFace dataset object...")
        great_ds = TaptapDataset.from_pandas(df)
        great_ds.set_args(
            numerical_features=numerical_features,
            target=target_col,
            numerical_modeling=self.numerical_modeling,
        )
        great_ds.set_tokenizer(self.tokenizer)

        # Set training hyperparameters
        logging.info("Create Taptap Trainer...")
        training_args = TrainingArguments(self.experiment_dir,
                                          max_steps=self.steps,
                                          per_device_train_batch_size=self.batch_size,
                                          save_steps=self.steps,
                                          # lr_scheduler_type='constant',
                                          **self.train_hyperparameters)
        great_trainer = TaptapTrainer(self.model, training_args, train_dataset=great_ds, tokenizer=self.tokenizer,
                                      data_collator=TaptapDataCollator(self.tokenizer))

        logging.info("Start training...")
        great_trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        return great_trainer

    def sample(self,
               n_samples: int,
               data: pd.DataFrame,
               task: str,
               constrain_dist: bool=False,
               start_col: tp.Optional[str] = "",
               start_col_dist: tp.Optional[tp.Union[dict, list]] = None,
               temperature: float = 0.7, k: int = 100,
               max_length: int = 1024, device: str = "cuda",
               imbalance: bool = False
               ) -> pd.DataFrame or (pd.DataFrame, pd.DataFrame):
        """ Generate synthetic tabular data samples

        Args:
            n_samples: Number of synthetic samples to generate
            start_col: Feature to use as starting point for the generation process. If not given, the target
             learned during the fitting is used as starting point
            start_col_dist: Feature distribution of the starting feature. Should have the format
             "{F1: p1, F2: p2, ...}" for discrete columns or be a list of possible values for continuous columns.
             If not given, the target distribution learned during the fitting is used as starting point
            temperature: The generation samples each token from the probability distribution given by a softmax
             function. The temperature parameter controls the softmax function. A low temperature makes it sharper
             (0 equals greedy search), a high temperature brings more diversity but also uncertainty into the output.
             See this blog article (https://huggingface.co/blog/how-to-generate) to read more about the generation
             process
            k: Sampling Batch Size. Set as high as possible. Speeds up the generation process significantly
            max_length: Maximal number of tokens to generate - has to be long enough to not cut any information!
            device: Set to "cpu" if the GPU should not be used. You can also specify the concrete GPU
            method: different methods for sampling. Choose from ['original', 'random', 'bidirectional']

        Returns:
            Pandas DataFrame with n_samples rows of generated data
        """
        self.tokenizer.padding_side = "left"
        great_start = self._get_start_sampler(start_col, start_col_dist)

        self.cat_dist = {}
        if constrain_dist is not False:
            for col in data.select_dtypes(exclude=np.number).columns.to_list():
                self.cat_dist[col] = _get_column_distribution(data, col, task)
        if self.numerical_modeling != 'original':
            numerical_features = data.select_dtypes(include=np.number).columns.to_list()
        else:
            numerical_features = []
        # Move model to device
        self.model.to(device)

        # Init empty DataFrame for the generated samples
        df_gen = pd.DataFrame(columns=self.columns)

        # Start generation process
        with tqdm(total=n_samples) as pbar:
            already_generated = 0
            while n_samples > df_gen.shape[0]:
                start = great_start.get_start_tokens(k, imbalance, self.numerical_modeling)

                # Generate tokens
                tokens = self.model.generate(input_ids=start["input_ids"].to(device),
                                             attention_mask=start["attention_mask"].to(device),

                                             max_length=max_length,
                                             do_sample=True, temperature=temperature, pad_token_id=50256,
                                             bad_words_ids=[[6045]])

                # Convert tokens back to tabular data
                text_data = _convert_tokens_to_text(tokens, self.tokenizer)
                df_gen = _convert_text_to_tabular_data(text_data, df_gen, self.cat_dist,
                                                       numerical_features=numerical_features,
                                                       numerical_modeling=self.numerical_modeling,
                                                       )
                # Remove rows with flawed numerical values
                for i_num_cols in self.num_cols:
                    df_gen[i_num_cols] = pd.to_numeric(df_gen[i_num_cols], errors='coerce')

                df_gen[self.num_cols] = df_gen[self.num_cols].astype(float)

                # Update process bar
                pbar.update(df_gen.shape[0] - already_generated)
                already_generated = df_gen.shape[0]

        df_gen = df_gen.reset_index(drop=True)
        self.tokenizer.padding_side = "right"
        return df_gen.head(n_samples)

    def impute(self, data: pd.DataFrame, temperature: float = 0.7, max_length: int = 1024,
                     k: int = 100, device: str = "cuda", do_sample=False):

        self.tokenizer.padding_side = "left"
        df_gen = None
        self.model.to(device)
        numerical_features = data.select_dtypes(include=np.number).columns.to_list()
        for i in range(3):
            print(f"Running {i} iteration.")
            if df_gen is not None:
                data = df_gen.copy()
            starting_prompt = []
            df_gen = [pd.DataFrame(columns=self.columns)]
            for idx, row in data.iterrows():
                if row.isna().any():
                    sentence = ', '.join([_get_string(self.numerical_modeling, numerical_features,
                                                      f, str(row.loc[f])) for f in row.index[::-1] if
                                          not pd.isna(row.loc[f])])
                    sentence += ','
                    starting_prompt.append(sentence)
                else:
                    df_tmp = pd.DataFrame(row).T
                    df_tmp.columns = self.columns
                    df_gen.append(df_tmp)
            df_gen = pd.concat(df_gen, axis=0)
            generated_data = []
            with tqdm(total=len(generated_data)) as pbar:
                while starting_prompt:
                    inputs = self.tokenizer(starting_prompt[:k], return_tensors="pt", padding=True)
                    # # Generate tokens
                    gen = self.model.generate(input_ids=inputs["input_ids"].to(device),
                                              attention_mask=inputs["attention_mask"].to(device),
                                              max_length=max_length,
                                              do_sample=do_sample, temperature=temperature,
                                              pad_token_id=50256, num_beams=3,
                                              length_penalty=-1,
                                              # force_words_ids=force_words_ids,
                                              # constraints=constraints,
                                              bad_words_ids=[[6045]])

                    # Convert Text back to Tabular Data
                    decoded_data = _convert_tokens_to_text(gen, self.tokenizer)
                    df_gen = _convert_text_to_tabular_data(decoded_data, df_gen,
                                                           numerical_features = numerical_features,
                                                           numerical_modeling = self.numerical_modeling
                    )
                    pbar.update(k)
                    starting_prompt = starting_prompt[k:]
            df_gen = _process_imputation(self.X_train_impute, df_gen)

            df_gen = df_gen.reset_index(drop=True)
        self.tokenizer.padding_side = "right"
        return df_gen


    def predict(self, data: pd.DataFrame, temperature: float = 0.7, max_length: int = 1024,
                     k: int = 100, device: str = "cuda", do_sample=False):
        self.tokenizer.padding_side = "left"
        df_gen = None
        self.model.to(device)
        numerical_features = data.select_dtypes(include=np.number).columns.to_list()
        for i in range(3):
            print(f"Running {i} iteration.")
            if df_gen is not None:
                data = df_gen.copy()
            starting_prompt = []
            df_gen = [pd.DataFrame(columns=self.columns)]
            for idx, row in data.iterrows():
                if row.isna().any():
                    sentence = ', '.join([_get_string(self.numerical_modeling, numerical_features,
                                                      f, str(row.loc[f])) for f in row.index[::-1] if
                                          not pd.isna(row.loc[f])])
                    sentence += ','
                    starting_prompt.append(sentence)
                else:
                    df_tmp = pd.DataFrame(row).T
                    df_tmp.columns = self.columns
                    df_gen.append(df_tmp)
            df_gen = pd.concat(df_gen, axis=0)
            generated_data = []
            with tqdm(total=len(generated_data)) as pbar:
                while starting_prompt:
                    inputs = self.tokenizer(starting_prompt[:k], return_tensors="pt", padding=True)
                    # # Generate tokens
                    gen = self.model.generate(input_ids=inputs["input_ids"].to(device),
                                              attention_mask=inputs["attention_mask"].to(device),
                                              max_length=max_length,
                                              do_sample=do_sample, temperature=temperature,
                                              pad_token_id=50256,
                                              length_penalty=-1,
                                              bad_words_ids=[[6045]])

                    # Convert Text back to Tabular Data
                    decoded_data = _convert_tokens_to_text(gen, self.tokenizer)
                    df_gen = _convert_text_to_tabular_data(decoded_data, df_gen,
                                                           numerical_features = numerical_features,
                                                           numerical_modeling = self.numerical_modeling
                    )
                    pbar.update(k)
                    starting_prompt = starting_prompt[k:]
            df_gen = _process_imputation(self.X_train_impute, df_gen)

            df_gen = df_gen.reset_index(drop=True)
        self.tokenizer.padding_side = "right"
        return df_gen



    def save(self, path: str):
        """

        Saves the model weights and a configuration file in the given directory.

        Args:
            path: Path where to save the model
        """
        # Make directory
        if os.path.isdir(path):
            warnings.warn(f"Directory {path} already exists and is overwritten now.")
        else:
            os.mkdir(path)

        # Save attributes
        with open(path + "/config.json", "w") as f:
            attributes = self.__dict__.copy()
            attributes.pop("tokenizer")
            attributes.pop("model")

            # NDArray is not JSON serializable and therefore has to be converted into a list.
            if isinstance(attributes["conditional_col_dist"], np.ndarray):
                attributes["conditional_col_dist"] = list(attributes["conditional_col_dist"])

            json.dump(attributes, f)

        # Save model weights
        torch.save(self.model.state_dict(), path + "/model.pt")

    def load_finetuned_model(self, path: str):
        """

        Load the weights of a fine-tuned large language model into the GReaT pipeline

        Args:
            path: Path to the fine-tuned model
        """
        self.model.load_state_dict(torch.load(path))

    @classmethod
    def load_from_dir(cls, path: str):
        """

        Load trained GReaT model from directory.

        Args:
            path: Directory where GReaT model is saved

        Returns:
            New instance of GReaT loaded from directory
        """
        assert os.path.isdir(path), f"Directory {path} does not exist."

        # Load attributes
        with open(path + "/config.json", "r") as f:
            attributes = json.load(f)

        # Create new be_great model instance
        great = cls(attributes["llm"])

        # Set all attributes
        for k, v in attributes.items():
            setattr(great, k, v)

        # Load model weights
        great.model.load_state_dict(torch.load(path + "/model.pt", map_location="cpu"))

        return great

    def _update_column_information(self, df: pd.DataFrame):
        # Update the column names (and numerical columns for some sanity checks after sampling)
        self.columns = df.columns.to_list()
        self.num_cols = df.select_dtypes(include=np.number).columns.to_list()

    def _update_conditional_information(self, df: pd.DataFrame, conditional_col: tp.Optional[str] = None, task=None):
        assert conditional_col is None or isinstance(conditional_col, str), \
            f"The column name has to be a string and not {type(conditional_col)}"
        assert conditional_col is None or conditional_col in df.columns, \
            f"The column name {conditional_col} is not in the feature names of the given dataset"

        # Take the distribution of the conditional column for a starting point in the generation process
        self.conditional_col = conditional_col if conditional_col else df.columns[-1]
        self.conditional_col_dist = _get_column_distribution(df, self.conditional_col, task)

    def _get_start_sampler(self, start_col: tp.Optional[str],
                           start_col_dist: tp.Optional[tp.Union[tp.Dict, tp.List]]) -> TaptapStart:
        if start_col and start_col_dist is None:
            raise ValueError(f"Start column {start_col} was given, but no corresponding distribution.")
        if start_col_dist is not None and not start_col:
            raise ValueError(f"Start column distribution {start_col} was given, the column name is missing.")

        assert start_col is None or isinstance(start_col, str), \
            f"The column name has to be a string and not {type(start_col)}"
        assert start_col_dist is None or isinstance(start_col_dist, dict) or isinstance(start_col_dist, list), \
            f"The distribution of the start column on has to be a list or a dict and not {type(start_col_dist)}"

        start_col = start_col if start_col else self.conditional_col
        start_col_dist = start_col_dist if start_col_dist else self.conditional_col_dist


        if isinstance(start_col_dist, dict):
            return CategoricalStart(self.tokenizer, start_col, start_col_dist)
        elif isinstance(start_col_dist, list):
            return ContinuousStart(self.tokenizer, start_col, start_col_dist)
        else:
            return RandomStart(self.tokenizer, self.columns)
