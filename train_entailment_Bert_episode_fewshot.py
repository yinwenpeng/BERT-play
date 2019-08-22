# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""

from __future__ import absolute_import, division, print_function

import argparse
import csv
import logging
import os
import random
import sys
import codecs
import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from torch.nn import CrossEntropyLoss, MSELoss
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score

from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_pretrained_bert.modeling import BertConfig, WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.modeling import BertForSequenceClassification_wenpeng_flexible as BertForSequenceClassification
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert import BertModel
from pytorch_pretrained_bert.optimization import BertAdam, warmup_linear

from bert_common_functions import store_bert_model

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines


class MrpcProcessor(DataProcessor):
    """Processor for the MRPC data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, "train.tsv")))
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[3]
            text_b = line[4]
            label = line[0]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class MnliProcessor(DataProcessor):
    """Processor for the MultiNLI data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev_matched.tsv")),
            "dev_matched")

    def get_labels(self):
        """See base class."""
        return ["contradiction", "entailment", "neutral"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[8]
            text_b = line[9]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class MnliMismatchedProcessor(MnliProcessor):
    """Processor for the MultiNLI Mismatched data set (GLUE version)."""

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev_mismatched.tsv")),
            "dev_matched")


class ColaProcessor(DataProcessor):
    """Processor for the CoLA data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[3]
            label = line[1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class Sst2Processor(DataProcessor):
    """Processor for the SST-2 data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[0]
            label = line[1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class StsbProcessor(DataProcessor):
    """Processor for the STS-B data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return [None]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[7]
            text_b = line[8]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class QqpProcessor(DataProcessor):
    """Processor for the STS-B data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            try:
                text_a = line[3]
                text_b = line[4]
                label = line[5]
            except IndexError:
                continue
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class QnliProcessor(DataProcessor):
    """Processor for the STS-B data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")),
            "dev_matched")

    def get_labels(self):
        """See base class."""
        return ["entailment", "not_entailment"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class RteProcessor(DataProcessor):
    """Processor for the RTE data set (GLUE version)."""

    def get_examples_SciTail_wenpeng(self, filename):
        '''
        can read the training file, dev and test file
        '''
        readfile = codecs.open(filename, 'r', 'utf-8')
        line_co=0
        examples=[]
        for row in readfile:
            line=row.strip().split('\t')
            if len(line)==3:
                guid = "train-"+str(line_co)
                text_a = line[0].strip()
                text_b = line[1].strip()
                label = 'entailment' if line[2] == 'entails' else 'not_entailment'
                examples.append(
                    InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
                line_co+=1
        readfile.close()
        print('loaded  size:', line_co)
        return examples

    def get_examples_SICK_wenpeng(self, filename):
        '''
        can read the training file, dev and test file
        '''
        readfile = codecs.open(filename, 'r', 'utf-8')
        line_co=0
        examples=[]
        for row in readfile:
            line=row.strip().split('\t')
            if len(line)==4:
                guid = "train-"+str(line_co)
                text_a = line[0].strip()
                text_b = line[1].strip()
                label = 'entailment' if line[2] == '1' else 'not_entailment'
                examples.append(
                    InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
                line_co+=1
        readfile.close()
        print('loaded  size:', line_co)
        return examples

    def get_examples_FEVER_wenpeng(self, filename):
        '''
        can read the training file, dev and test file
        '''
        readfile = codecs.open(filename, 'r', 'utf-8')
        line_co=0
        examples=[]
        for row in readfile:
            line=row.strip().split('\t')
            if len(line)==3:
                guid = "train-"+str(line_co)
                text_a = line[0].strip()
                text_b = line[1].strip()
                label = line[2].strip()
                examples.append(
                    InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
                line_co+=1
        readfile.close()
        print('loaded  size:', line_co)
        return examples

    def get_examples_MNLI_wenpeng(self, filename_list):
        '''
        can read the training file, dev and test file
        '''
        examples=[]
        for filename in filename_list:
            readfile = codecs.open(filename, 'r', 'utf-8')
            line_co=0
            for row in readfile:
                if line_co>0:
                    line=row.strip().split('\t')
                    guid = "train-"+str(line_co-1)
                    text_a = line[8].strip()
                    text_b = line[9].strip()
                    label = 'entailment' if line[-1] == 'entailment' else 'not_entailment'
                    examples.append(
                        InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
                line_co+=1
            readfile.close()
        print('loaded  size:', line_co)
        return examples

    def get_train_examples_wenpeng(self, filename):
        readfile = codecs.open(filename, 'r', 'utf-8')
        line_co=0
        examples=[]
        for row in readfile:
            if line_co>0:
                line=row.strip().split('\t')
                guid = "train-"+line[0]
                text_a = line[1]
                text_b = line[2]
                label = line[-1]
                examples.append(
                    InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
                line_co+=1
            else:
                line_co+=1
                continue
        readfile.close()
        print('loaded training size:', line_co)
        return examples

    def get_combined_train_examples_wenpeng(self, filename, filter_str):
        readfile = codecs.open(filename, 'r', 'utf-8')
        line_co=0
        examples=[]
        remove=0
        for row in readfile:

            line=row.strip().split('\t')
            if line[-1] !=filter_str:
                guid = "train-"+str(line_co)
                text_a = line[1]
                text_b = line[2]
                label = 'entailment' if line[0] == '1' else 'not_entailment'
                examples.append(
                    InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
                line_co+=1
            else:
                remove+=1

        readfile.close()
        print('loaded training size:', line_co, ' remove size:', remove)
        return examples

    def get_sequence_train_examples_wenpeng(self, filename, flag_strlist):
        flag2index= {flag: index for index, flag in enumerate(flag_strlist)}
        readfile = codecs.open(filename, 'r', 'utf-8')
        examples_sequence = [[] for i in range(len(flag_strlist))]
        size_sequence = [0]*len(flag_strlist)
        # examples=[]
        remove=0
        for row in readfile:

            line=row.strip().split('\t')
            index  = flag2index.get(line[-1].strip())
            if index is not None:
                guid = "train-"+str(size_sequence[index])
                text_a = line[1]
                text_b = line[2]
                label = 'entailment' if line[0] == '1' else 'not_entailment'
                examples_sequence[index].append(
                    InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
                size_sequence[index]+=1
        readfile.close()
        print('loaded training size:', flag_strlist, size_sequence, ' remove size:', remove)
        return examples_sequence

    def get_sequence_train_examples_and_split_pos_neg_wenpeng(self, filename, flag_strlist):
        '''
        in this function, we extract all the training examples, but we also store the Positive
        and negative examples separately for episode learning;
        for the last RTE task, we only load 10 for pos and neg
        '''
        flag2index= {flag: index for index, flag in enumerate(flag_strlist)}
        readfile = codecs.open(filename, 'r', 'utf-8')
        examples_sequence = [[] for i in range(len(flag_strlist))]
        pos_examples_sequence = [[] for i in range(len(flag_strlist))]
        neg_examples_sequence = [[] for i in range(len(flag_strlist))]

        size_sequence = [0]*len(flag_strlist)
        # examples=[]
        remove=0
        for row in readfile:

            line=row.strip().split('\t')
            index  = flag2index.get(line[-1].strip())
            if index is not None:
                if index == len(flag_strlist)-1:
                    #the target task, only load 10 pos and 10 neg
                    if size_sequence[index] < 20: #continue storing
                        label = 'entailment' if line[0] == '1' else 'not_entailment'
                        if label == 'entailment':
                            if len(pos_examples_sequence[index]) < 10:
                                guid = "train-"+str(size_sequence[index])
                                text_a = line[1]
                                text_b = line[2]
                                pos_examples_sequence[index].append(
                                    InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
                                size_sequence[index]+=1
                            else:
                                continue
                        else: # not_entailment
                            if len(neg_examples_sequence[index]) < 10:
                                guid = "train-"+str(size_sequence[index])
                                text_a = line[1]
                                text_b = line[2]
                                neg_examples_sequence[index].append(
                                    InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
                                size_sequence[index]+=1
                            else:
                                continue

                    else:
                        continue # next line


                else:

                    guid = "train-"+str(size_sequence[index])
                    text_a = line[1]
                    text_b = line[2]
                    label = 'entailment' if line[0] == '1' else 'not_entailment'
                    examples_sequence[index].append(
                        InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))

                    if label == 'entailment':
                        pos_examples_sequence[index].append(
                            InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
                    else:
                        neg_examples_sequence[index].append(
                            InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
                    size_sequence[index]+=1
        readfile.close()
        print('loaded training size:', flag_strlist, size_sequence, ' remove size:', remove)
        print('specific pos training sizes:', [len(seq) for seq in pos_examples_sequence], 'neg training sizes:', [len(seq) for seq in neg_examples_sequence])
        return examples_sequence, pos_examples_sequence, neg_examples_sequence

    def get_test_examples_wenpeng(self, filename):
        readfile = codecs.open(filename, 'r', 'utf-8')
        line_co=0
        examples=[]
        for row in readfile:
            line=row.strip().split('\t')
            if len(line)==3:
                guid = "test-"+str(line_co)
                text_a = line[1]
                text_b = line[2]
                label = 'entailment' if line[0] == '1' else 'not_entailment'
                examples.append(
                    InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
                line_co+=1

        readfile.close()
        print('loaded test size:', line_co)
        return examples

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["entailment", "not_entailment"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class WnliProcessor(DataProcessor):
    """Processor for the WNLI data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer, output_mode):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label : i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
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
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        if output_mode == "classification":
            label_id = label_map[example.label]
        elif output_mode == "regression":
            label_id = float(example.label)
        else:
            raise KeyError(output_mode)

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                    [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_id))
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds)
    return {
        "acc": acc,
        "f1": f1,
        "acc_and_f1": (acc + f1) / 2,
    }


def pearson_and_spearman(preds, labels):
    pearson_corr = pearsonr(preds, labels)[0]
    spearman_corr = spearmanr(preds, labels)[0]
    return {
        "pearson": pearson_corr,
        "spearmanr": spearman_corr,
        "corr": (pearson_corr + spearman_corr) / 2,
    }


def compute_metrics(task_name, preds, labels):
    assert len(preds) == len(labels)
    if task_name == "cola":
        return {"mcc": matthews_corrcoef(labels, preds)}
    elif task_name == "sst-2":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "mrpc":
        return acc_and_f1(preds, labels)
    elif task_name == "sts-b":
        return pearson_and_spearman(preds, labels)
    elif task_name == "qqp":
        return acc_and_f1(preds, labels)
    elif task_name == "mnli":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "mnli-mm":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "qnli":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "rte":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "wnli":
        return {"acc": simple_accuracy(preds, labels)}
    else:
        raise KeyError(task_name)


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                        "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                        "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the task to train.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--cache_dir",
                        default="",
                        type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--server_ip', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument('--flag', type=str, default='', help="Can be rename the stored BERT")
    parser.add_argument("--load_own_model",
                        action='store_true',
                        help="load_own_model.")
    args = parser.parse_args()

    # if args.server_ip and args.server_port:
    #     # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
    #     import ptvsd
    #     print("Waiting for debugger attach")
    #     ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
    #     ptvsd.wait_for_attach()

    processors = {
        "cola": ColaProcessor,
        "mnli": MnliProcessor,
        "mnli-mm": MnliMismatchedProcessor,
        "mrpc": MrpcProcessor,
        "sst-2": Sst2Processor,
        "sts-b": StsbProcessor,
        "qqp": QqpProcessor,
        "qnli": QnliProcessor,
        "rte": RteProcessor,
        "wnli": WnliProcessor,
    }

    output_modes = {
        "cola": "classification",
        "mnli": "classification",
        "mrpc": "classification",
        "sst-2": "classification",
        "sts-b": "regression",
        "qqp": "classification",
        "qnli": "classification",
        "rte": "classification",
        "wnli": "classification",
    }

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")


    task_name = args.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()
    output_mode = output_modes[task_name]

    label_list = processor.get_labels() #[0,1]
    num_labels = len(label_list)

    eval_examples = processor.get_test_examples_wenpeng('/home/wyin3/Datasets/RTE/test_RTE_1235.txt')
    # inter_task_names = ['SICK','SciTail', 'FEVER','MNLI','GLUE-RTE']
    inter_task_names = ['SciTail', 'SICK','GLUE-RTE']

    train_examples_sequence, train_pos_examples_sequence, train_neg_examples_sequence= processor.get_sequence_train_examples_and_split_pos_neg_wenpeng('/home/wyin3/Datasets/MNLI-SNLI-SciTail-RTE-SICK/all.6.train.txt', inter_task_names)
    # Prepare model
    # cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed_{}'.format(args.local_rank))
    # model = BertForSequenceClassification.from_pretrained(args.bert_model,cache_dir=cache_dir,num_labels=num_labels)

    model = BertModel.from_pretrained(args.bert_model)
    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
    print('\t\t\tload pretrained model succeed.........')
    model.to(device)

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]


    optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             t_total=30000)

    sample_train_size = 10
    sample_test_size = 10
    def examples_2_dataloader(examples, batch_size):
        '''
        to change: tokenizer
        '''
        train_features = convert_examples_to_features(
            examples, label_list, args.max_seq_length, tokenizer, output_mode)
        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)

        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        train_dataloader = DataLoader(train_data, sampler=RandomSampler(train_data), batch_size=batch_size)
        return train_dataloader
    def test_examples_2_dataloader(examples, batch_size):
        '''
        to change: tokenizer
        '''
        train_features = convert_examples_to_features(
            examples, label_list, args.max_seq_length, tokenizer, output_mode)
        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)

        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        train_dataloader = DataLoader(train_data, sampler=SequentialSampler(train_data), batch_size=batch_size)
        return train_dataloader, all_label_ids

    pos_10_batch_sequence_dataloader = [examples_2_dataloader(seq, sample_train_size) for seq in train_pos_examples_sequence[:-1]]
    pos_50_batch_sequence_dataloader = [examples_2_dataloader(seq, sample_test_size) for seq in train_pos_examples_sequence[:-1]]

    neg_10_batch_sequence_dataloader = [examples_2_dataloader(seq, sample_train_size) for seq in train_neg_examples_sequence[:-1]]
    neg_50_batch_sequence_dataloader = [examples_2_dataloader(seq, sample_test_size) for seq in train_neg_examples_sequence[:-1]]

    target_train_pos_few_dataloader = examples_2_dataloader(train_pos_examples_sequence[-1], sample_test_size)
    target_train_neg_few_dataloader = examples_2_dataloader(train_neg_examples_sequence[-1], sample_test_size)

    eval_dataloader, eval_all_label_ids = test_examples_2_dataloader(eval_examples, sample_train_size)
    max_test_acc = 0.0

    def batch_reps_2_batch_cos(source_reps, labels_reps):
        '''
        source_reps: (batch, hidden)
        labels_reps: (hidden, 2)
        return: #(batch_size, 2)
        '''
        target_norm = torch.norm(source_reps, dim=1) #batch_size
        label_norm = torch.norm(labels_reps, dim=0) #2
        norm_matrix = torch.mm(target_norm.view(-1,1), label_norm.view(1,-1))
        dot_matrix = torch.mm(source_reps, labels_reps)/(1e-6+norm_matrix) #(batch_size, 2)
        return dot_matrix

    global_step = 0
    nb_tr_steps = 0
    tr_loss = 0
    loss_fct = CrossEntropyLoss()
    iter_co=0
    while True:
        model.train()
        '''
        build proptype_emb
        '''
        pos_label_proptype_emb_cross_tasks = []
        neg_label_proptype_emb_cross_tasks = []
        for i in range(len(inter_task_names)-1):
            '''
            pos 10
            '''
            target_batch_index = random.randint(0,int(len(train_pos_examples_sequence[i])/sample_train_size))
            assert target_batch_index < len(pos_10_batch_sequence_dataloader[i])
            for target_size, target_batch in enumerate(pos_10_batch_sequence_dataloader[i]):#tqdm(train_dataloader, desc="episode")):
                if target_size == target_batch_index:
                    # print('\n\t find the random target batch...', target_batch_index, '\n')
                    target_batch = tuple(t.to(device) for t in target_batch)
                    target_input_ids, target_input_mask, target_segment_ids, target_label_ids = target_batch
                    _,target_batch_reps = model(target_input_ids, target_segment_ids, target_input_mask)
                    break
            pos_label_proptype_emb = torch.mean(target_batch_reps, dim=0, keepdim =True) #(1,hidden size)
            pos_label_proptype_emb_cross_tasks.append(pos_label_proptype_emb)
            '''
            neg 10
            '''
            target_batch_index = random.randint(0,int(len(train_neg_examples_sequence[i])/sample_train_size))
            assert target_batch_index < len(neg_10_batch_sequence_dataloader[i])
            for target_size, target_batch in enumerate(neg_10_batch_sequence_dataloader[i]):#tqdm(train_dataloader, desc="episode")):
                if target_size == target_batch_index:
                    # print('\n\t find the random target batch...', target_batch_index, '\n')
                    target_batch = tuple(t.to(device) for t in target_batch)
                    target_input_ids, target_input_mask, target_segment_ids, target_label_ids = target_batch
                    _,target_batch_reps = model(target_input_ids, target_segment_ids, target_input_mask)
                    break
            neg_label_proptype_emb = torch.mean(target_batch_reps, dim=0, keepdim =True) #(1,hidden size)
            neg_label_proptype_emb_cross_tasks.append(neg_label_proptype_emb)


        composed_pos_label_proptype_emb = torch.mean(torch.cat(tuple(pos_label_proptype_emb_cross_tasks), 0),dim=0, keepdim =True)#(1,hidden size)
        composed_neg_label_proptype_emb = torch.mean(torch.cat(tuple(neg_label_proptype_emb_cross_tasks), 0),dim=0, keepdim =True)#(1,hidden size)
        composed_labels_proptype_emb = torch.cat((composed_pos_label_proptype_emb, composed_neg_label_proptype_emb), dim=0)#(2,hidden size)
        '''
        train for multi-episode
        '''
        loss=0.0
        for i in range(len(inter_task_names)-1):

            '''
            pos 50
            '''
            target_batch_index = random.randint(0,int(len(train_pos_examples_sequence[i])/sample_test_size))
            assert target_batch_index < len(pos_50_batch_sequence_dataloader[i])
            for target_size, target_batch in enumerate(pos_50_batch_sequence_dataloader[i]):#tqdm(train_dataloader, desc="episode")):
                if target_size == target_batch_index:
                    # print('\n\t find the random target batch...', target_batch_index, '\n')
                    target_batch = tuple(t.to(device) for t in target_batch)
                    target_input_ids, target_input_mask, target_segment_ids, target_pos_label_ids = target_batch
                    _,pos_batch_reps = model(target_input_ids, target_segment_ids, target_input_mask)
                    break
            # pos_batch_reps = target_batch_reps #(batch, hidden)
            pos_2_label_matrix = batch_reps_2_batch_cos(pos_batch_reps, torch.t(composed_labels_proptype_emb))
            '''
            neg 50
            '''
            target_batch_index = random.randint(0,int(len(train_neg_examples_sequence[i])/sample_test_size))
            assert target_batch_index < len(neg_50_batch_sequence_dataloader[i])
            for target_size, target_batch in enumerate(neg_50_batch_sequence_dataloader[i]):#tqdm(train_dataloader, desc="episode")):
                if target_size == target_batch_index:
                    # print('\n\t find the random target batch...', target_batch_index, '\n')
                    target_batch = tuple(t.to(device) for t in target_batch)
                    target_input_ids, target_input_mask, target_segment_ids, target_neg_label_ids = target_batch
                    _,neg_batch_reps = model(target_input_ids, target_segment_ids, target_input_mask)
                    break
            # neg_batch_reps = target_batch_reps
            neg_2_label_matrix = batch_reps_2_batch_cos(neg_batch_reps, torch.t(composed_labels_proptype_emb))

            logits = torch.cat((pos_2_label_matrix,neg_2_label_matrix),dim=0) #(50*2, 2)
            target_compose_label_ids = torch.cat((target_pos_label_ids.view(-1),target_neg_label_ids.view(-1)),dim=0) #2*batch


            loss_i = loss_fct(logits.view(-1, num_labels), target_compose_label_ids)
            loss+=loss_i

        loss.backward(retain_graph=True)
        optimizer.step()
        optimizer.zero_grad()
        '''
        fine-tune on target episode
        '''
        model.train()
        target_preds = []
        target_ids = []
        target_list_pos_propotypes = []
        for target_size, target_batch in enumerate(target_train_pos_few_dataloader):#tqdm(train_dataloader, desc="episode")):
            target_batch = tuple(t.to(device) for t in target_batch)
            target_input_ids, target_input_mask, target_segment_ids, target_pos_label_ids = target_batch
            _,pos_batch_reps = model(target_input_ids, target_segment_ids, target_input_mask)
            pos_2_label_matrix = batch_reps_2_batch_cos(pos_batch_reps, torch.t(composed_labels_proptype_emb))
            target_preds.append(pos_2_label_matrix)
            target_ids.append(target_pos_label_ids.view(-1))
            target_list_pos_propotypes.append( torch.mean(pos_batch_reps, dim=0, keepdim =True))
        target_composed_pos_label_proptype_emb = torch.mean(torch.cat(tuple(target_list_pos_propotypes), 0),dim=0, keepdim =True)

        target_list_neg_propotypes = []
        for target_size, target_batch in enumerate(target_train_neg_few_dataloader):#tqdm(train_dataloader, desc="episode")):
            target_batch = tuple(t.to(device) for t in target_batch)
            target_input_ids, target_input_mask, target_segment_ids, target_pos_label_ids = target_batch
            _,neg_batch_reps = model(target_input_ids, target_segment_ids, target_input_mask)
            neg_2_label_matrix = batch_reps_2_batch_cos(neg_batch_reps, torch.t(composed_labels_proptype_emb))
            target_preds.append(pos_2_label_matrix)
            target_ids.append(target_pos_label_ids.view(-1))
            target_list_neg_propotypes.append( torch.mean(neg_batch_reps, dim=0, keepdim =True))
        target_composed_neg_label_proptype_emb = torch.mean(torch.cat(tuple(target_list_neg_propotypes), 0),dim=0, keepdim =True)
        target_composed_labels_proptype_emb = torch.cat((target_composed_pos_label_proptype_emb, target_composed_neg_label_proptype_emb), dim=0)#(2,hidden size)
        '''replace the composed label emb'''
        # composed_labels_proptype_emb = (composed_labels_proptype_emb+target_composed_labels_proptype_emb)/2.0

        fine_tune_logits = torch.cat(tuple(target_preds),dim=0)
        fine_tune_label_ids = torch.cat(tuple(target_ids),dim=0)
        fine_tune_loss = loss_fct(fine_tune_logits.view(-1, num_labels), fine_tune_label_ids)
        fine_tune_loss.backward()
        optimizer.step()
        optimizer.zero_grad()


        iter_co+=1
        print('iter_co:', iter_co)
        if iter_co% 100==0:
            '''
            evaluate after each epoch
            '''
            model.eval()
            eval_loss = 0
            nb_eval_steps = 0
            preds = []

            for input_ids, input_mask, segment_ids, label_ids in tqdm(eval_dataloader, desc="Evaluating"):
                input_ids = input_ids.to(device)
                input_mask = input_mask.to(device)
                segment_ids = segment_ids.to(device)
                label_ids = label_ids.to(device)

                with torch.no_grad():
                    _, test_batch_reps = model(input_ids, segment_ids, input_mask)
                logits = batch_reps_2_batch_cos(test_batch_reps, torch.t(composed_labels_proptype_emb))
                if len(preds) == 0:
                    preds.append(logits.detach().cpu().numpy())
                else:
                    preds[0] = np.append(
                        preds[0], logits.detach().cpu().numpy(), axis=0)

            preds = preds[0]
            preds = np.argmax(preds, axis=1)

            result = compute_metrics(task_name, preds, eval_all_label_ids.numpy())
            test_acc = result.get("acc")
            if test_acc > max_test_acc:
                max_test_acc = test_acc
                '''
                store the model
                '''
                # store_bert_model(model, tokenizer.vocab, args.output_dir, args.flag)
            print('test acc:', test_acc, ' max_test_acc:', max_test_acc)



if __name__ == "__main__":
    main()
# CUDA_VISIBLE_DEVICES=2 python -u train_entailment_Bert_episode_fewshot.py --task_name rte --do_train --do_lower_case --bert_model bert-base-uncased --max_seq_length 128 --train_batch_size 10 --eval_batch_size 10 --learning_rate 2e-5 --num_train_epochs 3 --data_dir '' --output_dir '/home/wyin3/Datasets/fine_tune_Bert_stored' --flag 'FineTuneOnFun'
