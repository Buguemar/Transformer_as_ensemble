import argparse
import glob
import logging
import os
import pickle
import random
import re
import shutil
import pandas as pd
from sklearn.metrics import f1_score, matthews_corrcoef, accuracy_score
import time
import random
import datetime
import seaborn as sns
from nltk.tokenize import word_tokenize
import transformers
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from transformers import (
    WEIGHTS_NAME,
    AdamW,
    BertConfig,
    BertForMaskedLM,
    BertForSequenceClassification,
    BertTokenizer,
    CamembertConfig,
    CamembertForMaskedLM,
    CamembertTokenizer,
    DistilBertConfig,
    DistilBertForMaskedLM,
    DistilBertTokenizer,
    get_linear_schedule_with_warmup,
    GPT2Config,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    OpenAIGPTConfig,
    OpenAIGPTLMHeadModel,
    OpenAIGPTTokenizer,
    RobertaConfig,
    RobertaForMaskedLM,
    RobertaTokenizer,
    get_linear_schedule_with_warmup,
)

from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split


def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def flat_fscore(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return f1_score(labels_flat, pred_flat, average='macro')


def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))


def make_padding_and_masks(max_len, tokenizer, input_ids):
    print('\nPadding/truncating all sentences to %d values...' % max_len)
    print('\nPadding token: "{:}", ID: {:}'.format(tokenizer.pad_token, tokenizer.pad_token_id))

    # Pad our input tokens with value 0.
    input_ids = pad_sequences(input_ids, maxlen=max_len, dtype="long", value=0, truncating="post", padding="post")

    print('Completado.')

    attention_masks = []

    # For each sentence...
    for sent in input_ids:
        att_mask = [int(token_id > 0) for token_id in sent]
        attention_masks.append(att_mask)

    return input_ids, attention_masks

def data_batches(input_ids_new, attention_masks, n_labels, test_size=0.1, batch_size=16, mode='train'):
    if mode=='train':
        # Use 90% for training and 10% for validation.
        train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(input_ids_new, n_labels,
                                                                                            random_state=2018, 
                                                                                            test_size= test_size)
        # Do the same for the masks.
        train_masks, validation_masks, _, _ = train_test_split(attention_masks, n_labels, random_state=2018, 
                                                               test_size=test_size)

        train_inputs = torch.tensor(train_inputs)
        validation_inputs = torch.tensor(validation_inputs)
        train_labels = torch.tensor(train_labels)
        validation_labels = torch.tensor(validation_labels)
        train_masks = torch.tensor(train_masks)
        validation_masks = torch.tensor(validation_masks)

        # For fine-tuning BERT on a specific task, the authors recommend a batch size of
        # 16 or 32
        # Create the DataLoader for our training set.
        train_data = TensorDataset(train_inputs, train_masks, train_labels)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
        train_all = (train_data, train_sampler, train_dataloader)
        # Create the DataLoader for our validation set.
        validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
        validation_sampler = SequentialSampler(validation_data)
        validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)
        val_all= (validation_data, validation_sampler, validation_dataloader)
        return train_all, val_all
        
    else:  #EVAL
        # Convert to tensors.
        prediction_inputs = torch.tensor(input_ids_new)
        prediction_masks = torch.tensor(attention_masks)
        prediction_labels = torch.tensor(n_labels)

        # Create the DataLoader.
        prediction_data = TensorDataset(prediction_inputs, prediction_masks, prediction_labels)
        prediction_sampler = SequentialSampler(prediction_data)
        prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)
        eval_all = (prediction_data, prediction_sampler, prediction_dataloader)
        return eval_all


def save_model_to(output_dir, model, tokenizer, state_dict):
    # Create output directory if needed
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print("Saving model to %s" % output_dir)
    # Save a trained model, configuration and tokenizer using `save_pretrained()`.
    # They can then be reloaded using `from_pretrained()`
    model_to_save = model.module if hasattr(model, 'module') else model
    model_to_save.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    torch.save(state_dict, os.path.join(output_dir, 'training_and_results_args.bin'))
