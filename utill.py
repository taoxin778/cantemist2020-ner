import logging
import os
from enum import Enum
from typing import List, Optional, Union
from transformers import PreTrainedTokenizer
from dataclasses import dataclass
import pandas as pd
import torch
from torch import nn
from torch.utils.data.dataset import Dataset, TensorDataset

from data_proceess import get_doc_name
logger = logging.getLogger(__name__)


@dataclass
class InputExample:
    """
    A single training/test example for token classification.

    Args:
        guid: Unique id for the example.
        words: list. The words of the sequence.
        labels: (Optional) list. The labels for each word of the sequence. This should be
        specified for train and dev examples, but not for test examples.
    """

    guid: str
    words: List[str]
    labels: Optional[List[str]]
    spacys: List


@dataclass
class InputFeatures:
    """
    A single set of features of data.
    Property names are the same names as the corresponding inputs to a model.
    """

    input_ids: List[int]
    attention_mask: List[int]
    token_type_ids: Optional[List[int]] = None
    label_ids: Optional[List[int]] = None


class Split(Enum):
    train = "train"
    dev = "dev"
    test = "test"


class NerDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach
    soon.
    """

    features: List[InputFeatures]
    pad_token_label_id: int = nn.CrossEntropyLoss().ignore_index

    # Use cross entropy ignore_index as padding label id so that only
    # real label ids contribute to the loss later.

    def __init__(
            self,
            data_dir: str,
            tokenizer: PreTrainedTokenizer,
            labels: List[str],
            model_type: str,
            max_seq_length: Optional[int] = None,
            overwrite_cache=False,
            mode: Split = Split.train,
    ):
        examples = read_examples_from_file(data_dir, mode)

        self.features = convert_examples_to_features(
            examples,
            labels,
            max_seq_length,
            tokenizer,
            cls_token_at_end=bool(model_type in ["xlnet"]),
            # xlnet has a cls token at the end
            cls_token=tokenizer.cls_token,
            cls_token_segment_id=2 if model_type in ["xlnet"] else 0,
            sep_token=tokenizer.sep_token,
            sep_token_extra=False,
            # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
            pad_on_left=bool(tokenizer.padding_side == "left"),
            pad_token=tokenizer.pad_token_id,
            pad_token_segment_id=tokenizer.pad_token_type_id,
            pad_token_label_id=self.pad_token_label_id,
        )


def read_examples_from_file(data_dict) -> List[InputExample]:
    exampls = []
    for item in data_dict:
        exampl = InputExample(words=item[2], guid=item[0], labels=item[3], spacys=item[1])
        exampls.append(exampl)
    return exampls


def convert_examples_to_features(
        examples: List[InputExample],
        label_list: List[str],
        max_seq_length: int,
        tokenizer: PreTrainedTokenizer,
        cls_token_at_end=False,
        cls_token="[CLS]",
        cls_token_segment_id=0,
        sep_token="[SEP]",
        sep_token_extra=False,
        pad_on_left=False,
        pad_token=0,
        pad_token_segment_id=0,
        pad_token_label_id=-100,
        sequence_a_segment_id=0,
        mask_padding_with_zero=True,
) -> List[InputFeatures]:
    """ Loads a data file into a list of `InputFeatures`
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """
    # TODO clean up all this to leverage built-in features of tokenizers

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10_000 == 0:
            logger.info("Writing example %d of %d", ex_index, len(examples))

        tokens = []
        label_ids = []
        for word, label in zip(example.words, example.labels):
            word_tokens = tokenizer.tokenize(word)

            # bert-base-multilingual-cased sometimes output "nothing ([]) when calling tokenize with just a space.
            if len(word_tokens) > 0:
                tokens.extend(word_tokens)
                # Use the real label id for the first token of the word, and padding ids for the remaining tokens
                label_ids.extend([label_map[label]] + [pad_token_label_id] * (len(word_tokens) - 1))

        # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
        special_tokens_count = tokenizer.num_special_tokens_to_add()
        if len(tokens) > max_seq_length - special_tokens_count:
            tokens = tokens[: (max_seq_length - special_tokens_count)]
            label_ids = label_ids[: (max_seq_length - special_tokens_count)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens += [sep_token]
        label_ids += [pad_token_label_id]
        if sep_token_extra:
            # roberta uses an extra separator b/w pairs of sentences
            tokens += [sep_token]
            label_ids += [pad_token_label_id]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        if cls_token_at_end:
            tokens += [cls_token]
            label_ids += [pad_token_label_id]
            segment_ids += [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            label_ids = [pad_token_label_id] + label_ids
            segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
            label_ids = ([pad_token_label_id] * padding_length) + label_ids
        else:
            input_ids += [pad_token] * padding_length
            input_mask += [0 if mask_padding_with_zero else 1] * padding_length
            segment_ids += [pad_token_segment_id] * padding_length
            label_ids += [pad_token_label_id] * padding_length

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length     
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length

        # if ex_index < 5:
        #     logger.info("*** Example ***")
        #     logger.info("guid: %s", example.guid)
        #     logger.info("tokens: %s", " ".join([str(x) for x in tokens]))
        #     logger.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
        #     logger.info("input_mask: %s", " ".join([str(x) for x in input_mask]))
        #     logger.info("segment_ids: %s", " ".join([str(x) for x in segment_ids]))
        #     logger.info("label_ids: %s", " ".join([str(x) for x in label_ids]))

        if "token_type_ids" not in tokenizer.model_input_names:
            segment_ids = None

        features.append(
            InputFeatures(
                input_ids=input_ids, attention_mask=input_mask, token_type_ids=segment_ids, label_ids=label_ids
            )
        )
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    return dataset


def get_cancer_span(pred_tags, examples):
    assert len(examples) == len(pred_tags)
    entity_list = list()
    tag_temp = []
    token_temp = []
    for example, tags in zip(examples, pred_tags):
        prev_tag = None
        full_flag = False
        id = example.guid
        tokens = example.spacys
        # if id == "doc_test_1009":
        #     print("1009")
        for token, tag in zip(tokens, tags):
            if "B" in tag:
                if len(tag_temp) != 0:
                    entity_list.append((id, tag_temp, token_temp))
                    tag_temp = []
                    token_temp = []
                tag_temp.append(tag)
                token_temp.append(token)
                prev_tag = "B"
            elif "I" in tag:
                if prev_tag == "B" or prev_tag == "I":
                    token_temp.append(token)
                    tag_temp.append(tag)
                else:
                    if len(tag_temp) != 0:
                        entity_list.append((id, tag_temp, token_temp))
                    tag_temp = [tag]
                    token_temp = [token]
                prev_tag = "I"
            else:
                prev_tag = "O"
    return entity_list


def write2file(entities, answer_file):
    file_name_set = {entity[0] for entity in entities}
    entity_dict = {filename: [] for filename in file_name_set}
    for ent in entities:
        entity_dict[ent[0]].append(ent)
    for file, ents in entity_dict.items():
        path = os.path.join(answer_file, file + ".ann")
        with open(path, "w", encoding="utf-8")as f:
            for idx, ent in enumerate(ents):
                num = "T{}".format(idx + 1)
                s1 = ent[2][0].idx
                s2 = ent[2][-1].idx + len(ent[2][-1])
                type_span = "MORFOLOGIA_NEOPLASIA {} {}".format(s1, s2)
                text = " ".join([t.text for t in ent[2]])
                f.write(num + "\t" + type_span + "\t" + text + "\n")




