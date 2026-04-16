from pathlib import Path

import pandas as pd
import torch
from loguru import logger
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import TensorDataset as TorchTensorDataset
from tqdm import tqdm
from transformers.models.auto.tokenization_auto import AutoTokenizer

from src.configs.constants import (
    BINARY_P_AND_Q_TASKS,
    BINARY_PARAGRAPH_ONLY_TASKS,
    DLModelNames,
    REGRESSION_PARAGRAPH_ONLY_TASKS,
    Fields,
    PredMode,
)
from src.configs.main_config import Args
from src.configs.models.base_model import DLModelArgs, MLModelArgs
from src.data.utils import load_raw_data


class TextDataSet(TorchDataset):
    """
    A PyTorch dataset for text data.
    """

    def __init__(self, cfg: Args):
        self.prediction_mode = cfg.data.task
        valid_modes = (
            BINARY_P_AND_Q_TASKS
            + BINARY_PARAGRAPH_ONLY_TASKS
            + REGRESSION_PARAGRAPH_ONLY_TASKS
        )
        if self.prediction_mode not in valid_modes:
            raise ValueError(
                f'Invalid value for PREDICTION_MODE: {self.prediction_mode}'
            )
        self.max_data_seq_len = cfg.data.max_seq_len
        self.max_model_supported_len = cfg.model.max_supported_seq_len
        self.actual_max_needed_len = min(
            self.max_data_seq_len, self.max_model_supported_len
        )
        self.num_special_tokens_to_add = cfg.model.num_special_tokens_add
        self.actual_max_seq_len = 0
        self.max_q_len = cfg.data.max_q_len
        assert isinstance(cfg.model, (DLModelArgs, MLModelArgs))
        self.prepend_eye_features_to_text = cfg.model.prepend_eye_features_to_text
        self.text_key_field = cfg.data.unique_trial_id_column
        self.preorder = cfg.model.preorder
        self.base_model_name = cfg.model.base_model_name

        self.print_tokens = True
        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.model.backbone,  # type: ignore
            is_split_into_words=True,
            add_prefix_space=True,
        )
        eye_token = '<eye>'
        self.tokenizer.add_special_tokens(
            special_tokens_dict={'additional_special_tokens': [eye_token]},
            replace_additional_special_tokens=False,
        )
        self.eye_token_id: int = self.tokenizer.convert_tokens_to_ids(eye_token)

        text_data = self.prepare_text_data(data_path=cfg.data.ia_path)
        # create a dict mapping from key column (as the dict key) to index (as the dict value)
        text_keys = text_data[self.text_key_field].copy()
        self.key_to_index = dict(zip(text_keys, text_keys.index))

        (
            self.text_features,
            self.inversions_lists,
        ) = self.convert_examples_to_features(
            text_data,
        )

        self.text_data = text_data

    def prepare_text_data(self, data_path: Path) -> pd.DataFrame:
        """
        Prepares the text data by loading it from a CSV file and selecting relevant columns.
        Args:
            data_path (Path): The path to the CSV file containing the text data.

        Returns:
            pd.DataFrame: A DataFrame containing the selected columns from the CSV file
                after dropping duplicates.
        """
        usecols = [
            field.value
            for field in [
                Fields.UNIQUE_TRIAL_ID,
                Fields.QUESTION,
                Fields.PARAGRAPH,
                Fields.CLAIM,
                Fields.CONTEXT,
                Fields.CLAIM_START_WORD_IDX,
                Fields.CLAIM_END_WORD_IDX,
                Fields.CONTEXT_START_WORD_IDX,
                Fields.CONTEXT_END_WORD_IDX,
            ]
        ]

        text_data = load_raw_data(data_path)

        missing_columns = [col for col in usecols if col not in text_data.columns]
        if missing_columns:
            logger.warning(f'Missing columns: {missing_columns}')
        existing_columns = [col for col in usecols if col in text_data.columns]
        logger.info(f'Using columns: {existing_columns}')

        text_data = text_data[existing_columns].copy()
        text_data = text_data.drop_duplicates(subset=self.text_key_field).reset_index(
            drop=True
        )
        return text_data

    def __len__(self) -> int:
        return len(self.key_to_index)

    def __getitem__(self, index: int) -> tuple[tuple[torch.Tensor, ...], list[int]]:
        features = self.text_features[index]
        inversions_list = self.inversions_lists[index]
        return features, inversions_list

    def convert_examples_to_features(
        self,
        examples: pd.DataFrame,
    ) -> tuple[torch.Tensor | TorchTensorDataset, list[list[int]]]:
        # Roberta tokenization
        """Loads a data file into a list of `InputBatch`s."""

        # we will use the formatting proposed in "Improving Language
        # Understanding by Generative Pre-Training" and suggested by
        # @jacobdevlin-google in this issue
        # https://github.com/google-research/bert/issues/38.
        assert self.tokenizer.sep_token_id is not None
        assert self.tokenizer.cls_token_id is not None
        paragraphs_input_ids_list = []
        paragraphs_masks_list = []
        input_ids_list: list[list[int] | list[list[int]]] = []
        input_masks_list: list[list[int] | list[list[int]]] = []
        claim_token_masks_list: list[list[int]] = []
        context_token_masks_list: list[list[int]] = []
        passages_length = []
        inversions_list = []
        full_lengths = []
        for example in tqdm(
            examples.itertuples(),
            total=len(examples),
            desc='Tokenizing',
        ):
            paragraph_ids, inversions, full_length, word_offset = self.tokenize(
                text=example.paragraph
            )
            full_lengths.append(full_length)
            # TODO Low priority: refactor to avoid duplication of input_ids and p_input_ids
            p_input_ids = paragraph_ids.copy()

            p_input_ids.insert(0, self.tokenizer.cls_token_id)

            # Zero-pad up to the sequence length.
            p_input_mask = [1] * len(p_input_ids) + [0] * (
                self.actual_max_needed_len - len(p_input_ids)
            )
            p_input_ids = p_input_ids + [1] * (
                self.actual_max_needed_len - len(p_input_ids)
            )

            # Add the paragraph to the lists
            paragraphs_input_ids_list.append(p_input_ids)
            paragraphs_masks_list.append(p_input_mask)

            if self.use_claim_context_input(example):
                (
                    input_ids,
                    input_masks,
                    claim_token_mask,
                    context_token_mask,
                    input_inversions,
                ) = self.process_claim_context_example(example)
            else:
                endings_ids = self.add_tokenized_question_if_needed(example)
                full_ending_ids = []
                for ending_tokens in endings_ids:
                    full_ending_ids.extend(
                        ending_tokens
                    )  # * If adding more than one ending, concatenate them. Consider adding separators.

                input_ids, input_masks = self.process_example(
                    paragraph_ids, full_ending_ids
                )
                claim_token_mask, context_token_mask = self.build_claim_context_masks(
                    example=example,
                    inversions=inversions,
                    paragraph_token_count=len(paragraph_ids),
                    word_offset=word_offset,
                    input_length=len(input_ids),
                )
                input_inversions = inversions

            input_ids_list.append(input_ids)
            input_masks_list.append(input_masks)
            claim_token_masks_list.append(claim_token_mask)
            context_token_masks_list.append(context_token_mask)

            if self.print_tokens:
                if isinstance(input_ids_list[0][0], list):
                    for ids in input_ids_list[0]:
                        logger.info(self.tokenizer.convert_ids_to_tokens(ids))
                else:
                    logger.info(self.tokenizer.convert_ids_to_tokens(input_ids_list[0]))
                self.print_tokens = False

            passages_length.append(len(paragraph_ids))
            inversions_list.append(input_inversions)
        if self.actual_max_needed_len > self.actual_max_seq_len:
            logger.warning(
                f'{self.actual_max_needed_len=} while max length in practice is {self.actual_max_seq_len}.'
            )

        features = TorchTensorDataset(
            torch.tensor(paragraphs_input_ids_list, dtype=torch.long),
            torch.tensor(paragraphs_masks_list, dtype=torch.long),
            torch.tensor(input_ids_list, dtype=torch.long),
            torch.tensor(input_masks_list, dtype=torch.long),
            torch.tensor(passages_length, dtype=torch.long),
            torch.tensor(full_lengths, dtype=torch.long),
            torch.tensor(claim_token_masks_list, dtype=torch.long),
            torch.tensor(context_token_masks_list, dtype=torch.long),
        )

        return features, inversions_list

    def build_inputs_with_special_tokens(
        self,
        context_ids: list[int],
        ending_ids: list[int],
    ) -> list[int]:
        """
        Based on from RobertaTokenizer.build_inputs_with_special_tokens
        #! Check where things break if making changes here
        """
        assert self.tokenizer.cls_token_id is not None
        assert self.tokenizer.sep_token_id is not None

        cls_token_id = self.tokenizer.cls_token_id
        sep_token_id = self.tokenizer.sep_token_id

        input_ids = [cls_token_id]

        if self.prepend_eye_features_to_text:
            input_ids.extend([self.eye_token_id, sep_token_id])

        input_ids += (
            context_ids + [sep_token_id, sep_token_id] + ending_ids + [sep_token_id]
        )
        return input_ids

    def process_example(
        self,
        paragraph_ids: list[int],
        ending_ids: list[int],
    ) -> tuple[list[int], list[int]]:
        input_ids = self.build_inputs_with_special_tokens(paragraph_ids, ending_ids)

        self.verify_input_length(input_ids)
        padding_length = self.actual_max_needed_len - len(input_ids)
        # Update input mask and padding for the concatenated sequence
        input_mask = [1] * len(input_ids) + [0] * padding_length
        padding_ids = [1] * padding_length  # 1 for roberta
        input_ids.extend(padding_ids)

        return input_ids, input_mask

    @staticmethod
    def _get_example_attr(example, field: Fields, default=None):
        return getattr(example, str(field), default)

    def use_claim_context_input(self, example) -> bool:
        if (
            self.prediction_mode != PredMode.CV
            or self.base_model_name != DLModelNames.CEC_GAZE_MODEL
        ):
            return False

        claim = self._get_example_attr(example, Fields.CLAIM, '')
        context = self._get_example_attr(example, Fields.CONTEXT, '')
        return bool(str(claim).strip()) and bool(str(context).strip())

    def tokenize_without_truncation(self, text: str) -> tuple[list[int], list[int]]:
        tokens = self.tokenizer(
            str(text).split(),
            is_split_into_words=True,
            add_special_tokens=False,
        )
        token_word_ids = tokens.word_ids()
        return tokens['input_ids'], [int(word_id) for word_id in token_word_ids]

    def process_claim_context_example(
        self,
        example,
    ) -> tuple[list[int], list[int], list[int], list[int], list[int]]:
        """Create [CLS] claim [SEP] [SEP] context [SEP] inputs for CEC-Gaze."""
        assert self.tokenizer.sep_token_id is not None
        assert self.tokenizer.cls_token_id is not None

        claim_text = str(self._get_example_attr(example, Fields.CLAIM, ''))
        context_text = str(self._get_example_attr(example, Fields.CONTEXT, ''))
        claim_start = int(
            self._get_example_attr(example, Fields.CLAIM_START_WORD_IDX, 0)
        )
        context_start = int(
            self._get_example_attr(example, Fields.CONTEXT_START_WORD_IDX, 0)
        )

        claim_ids, claim_word_ids = self.tokenize_without_truncation(claim_text)
        context_ids, context_word_ids = self.tokenize_without_truncation(context_text)
        claim_inversions = [claim_start + word_id for word_id in claim_word_ids]
        context_inversions = [context_start + word_id for word_id in context_word_ids]

        max_text_tokens = self.actual_max_needed_len - self.num_special_tokens_to_add
        if len(claim_ids) > max_text_tokens:
            claim_ids = claim_ids[-max_text_tokens:]
            claim_inversions = claim_inversions[-max_text_tokens:]
            context_ids = []
            context_inversions = []
        else:
            remaining_context_tokens = max_text_tokens - len(claim_ids)
            if remaining_context_tokens <= 0:
                context_ids = []
                context_inversions = []
            elif len(context_ids) > remaining_context_tokens:
                context_ids = context_ids[-remaining_context_tokens:]
                context_inversions = context_inversions[-remaining_context_tokens:]

        input_ids = [self.tokenizer.cls_token_id]
        claim_mask = [0]
        context_mask = [0]

        if self.prepend_eye_features_to_text:
            input_ids.extend([self.eye_token_id, self.tokenizer.sep_token_id])
            claim_mask.extend([0, 0])
            context_mask.extend([0, 0])

        input_ids.extend(claim_ids)
        claim_mask.extend([1] * len(claim_ids))
        context_mask.extend([0] * len(claim_ids))

        input_ids.extend([self.tokenizer.sep_token_id, self.tokenizer.sep_token_id])
        claim_mask.extend([0, 0])
        context_mask.extend([0, 0])

        input_ids.extend(context_ids)
        claim_mask.extend([0] * len(context_ids))
        context_mask.extend([1] * len(context_ids))

        input_ids.append(self.tokenizer.sep_token_id)
        claim_mask.append(0)
        context_mask.append(0)

        self.verify_input_length(input_ids)
        padding_length = self.actual_max_needed_len - len(input_ids)
        input_mask = [1] * len(input_ids) + [0] * padding_length
        input_ids.extend([1] * padding_length)
        claim_mask.extend([0] * padding_length)
        context_mask.extend([0] * padding_length)

        # One entry per non-CLS input position. -1 denotes separator/eye tokens
        # that should receive zero gaze features.
        input_inversions = (
            ([-1, -1] if self.prepend_eye_features_to_text else [])
            + claim_inversions
            + [-1, -1]
            + context_inversions
            + [-1]
        )

        return input_ids, input_mask, claim_mask, context_mask, input_inversions

    def add_tokenized_question_if_needed(
        self,
        example,
    ) -> list[list[int]]:
        """
        Processing of example endings based on prediction mode.
        """
        if self.prediction_mode in BINARY_P_AND_Q_TASKS:
            endings = [f'Question: {example.question}']
        else:
            endings = []

        endings_ids: list[list[int]] = [self.tokenize(ending)[0] for ending in endings]

        return endings_ids

    def build_claim_context_masks(
        self,
        example,
        inversions: list[int],
        paragraph_token_count: int,
        word_offset: int,
        input_length: int,
    ) -> tuple[list[int], list[int]]:
        """Build token masks for claim and context spans over the full model input."""

        claim_mask = [0] * input_length
        context_mask = [0] * input_length

        claim_start = getattr(example, Fields.CLAIM_START_WORD_IDX, None)
        claim_end = getattr(example, Fields.CLAIM_END_WORD_IDX, None)
        context_start = getattr(example, Fields.CONTEXT_START_WORD_IDX, None)
        context_end = getattr(example, Fields.CONTEXT_END_WORD_IDX, None)

        has_claim_span = (
            claim_start is not None
            and claim_end is not None
            and not pd.isna(claim_start)
            and not pd.isna(claim_end)
            and int(claim_end) > int(claim_start)
        )
        has_context_span = (
            context_start is not None
            and context_end is not None
            and not pd.isna(context_start)
            and not pd.isna(context_end)
            and int(context_end) > int(context_start)
        )

        token_offset = 3 if self.prepend_eye_features_to_text else 1
        for token_idx, shifted_word_idx in enumerate(inversions):
            model_token_idx = token_idx + token_offset
            absolute_word_idx = shifted_word_idx + word_offset

            if has_claim_span and int(claim_start) <= absolute_word_idx < int(claim_end):
                claim_mask[model_token_idx] = 1

            if has_context_span and int(context_start) <= absolute_word_idx < int(
                context_end
            ):
                context_mask[model_token_idx] = 1

        if sum(context_mask) == 0:
            end_idx = min(token_offset + paragraph_token_count, input_length)
            for token_idx in range(token_offset, end_idx):
                context_mask[token_idx] = 1

        return claim_mask, context_mask

    def verify_input_length(self, tokens: list[int]) -> None:
        assert len(tokens) <= self.actual_max_needed_len, (
            f'tokens length is {len(tokens)}, max_seq_length is {self.actual_max_needed_len}'
        )

        if len(tokens) > self.actual_max_seq_len:
            self.actual_max_seq_len = len(tokens)

    def tokenize(self, text: str) -> tuple[list[int], list[int], int, int]:
        """
        Tokenizes a paragraph into a list of tokens.
        If the tokenized text exceeds actual_max_needed_len, truncates to keep the last actual_max_needed_len tokens.

        Args:
            text (str): The paragraph to tokenize.

        Returns:
            tuple[list[str], list[int]]: The tokenized paragraph and the inversions list.

        """
        tokens = self.tokenizer(
            text.split(),
            is_split_into_words=True,
            add_special_tokens=False,
        )
        input_ids: list[int] = tokens['input_ids']
        token_word_ids: list[int] = tokens.word_ids()
        full_length = max(token_word_ids) + 1
        word_offset = 0
        # Truncate to actual_max_needed_len, keeping the last tokens
        max_length = (
            self.actual_max_needed_len - self.num_special_tokens_to_add - self.max_q_len
        )

        if len(input_ids) > max_length:
            input_ids = input_ids[-max_length:]
            token_word_ids = token_word_ids[-max_length:]
            min_id = min(token_word_ids)
            word_offset = min_id
            token_word_ids = [id_ - min_id for id_ in token_word_ids]

        return input_ids, token_word_ids, full_length, word_offset
