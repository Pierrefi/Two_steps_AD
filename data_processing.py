#!/usr/bin/env python3
# coding: utf-8
from __future__ import absolute_import, division, print_function


import json
import torch 
import logging 
import os 
import pickle 

import pandas as pd 
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from transformers import BertTokenizer, AutoTokenizer, RobertaTokenizer
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import warnings
warnings.filterwarnings('ignore')
from datasets import load_dataset
import random
import numpy as np 

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

random.seed(43)
batch_size = 32


def set_binary_label(dataframe, indomain, col='label'):
    if indomain:
        dataframe[col].values[:] = 0
    else:
        dataframe[col].values[:] = 1

def load_dataset_clinc(data_name, data_type='full'):
    with open('./dataset/CLINIC150/data_full.json', 'r') as f:
        data = json.load(f)
    field = "_".join(data_name.split("_")[1:])
    dataset = data[field]
    data_df = pd.DataFrame(dataset, columns=['text', 'label'])  # labels are not used for training
    return data_df

def load_extra_dataset(file_path ,drop_index=False, label=0):
    df = pd.read_csv(file_path, sep='\t', header=0)
    df['label'] = label
    df.rename(columns = {'sentence': 'text'}, inplace=True)
    if drop_index:
        df.drop(columns='index', inplace=True)
    df.dropna(inplace=True)
    return df

def encode_string_labels(df, label_col="label"):

    unique_labels = sorted(df[label_col].unique())
    label2id = {lab: i for i, lab in enumerate(unique_labels)}
    id2label = {i: lab for lab, i in label2id.items()}
    df[label_col] = df[label_col].map(label2id)

    return df

def create_datasets(dataset, id_labels, ood_labels, test_size=0.2, random_state=42, sample = False):
    train_df_base = pd.DataFrame(dataset['train'])
    
    if 'sentence' in train_df_base.columns:
        train_df_base.rename(columns={'sentence': 'text'}, inplace=True)
    if 'category' in train_df_base.columns:
        train_df_base.rename(columns={'category': 'label'}, inplace=True)
    train_df_base.drop(columns=[col for col in train_df_base.columns if col not in ['text', 'label']], inplace=True)

    
    if train_df_base['label'].dtype == object:
        train_df_base = encode_string_labels(train_df_base, "label")
    train_df_base, test_df_base = train_test_split(train_df_base, test_size=test_size, random_state=random_state)

    if sample is True:
        n_total = 10000
        labels = train_df_base["label"].unique()
        n_per_label = n_total // len(labels)

        train_df_base = (
            train_df_base
            .groupby("label", group_keys=False)
            .apply(lambda x: x.sample(
                n=min(len(x), n_per_label),
                random_state=42
            ))
            .reset_index(drop=True)
        )
        n_total_test = 10000
        labels_test = test_df_base["label"].unique()
        n_per_label = n_total_test // len(labels_test)

        test_df_base = (
            test_df_base
            .groupby("label", group_keys=False)
            .apply(lambda x: x.sample(
                n=min(len(x), n_per_label),
                random_state=42
            ))
            .reset_index(drop=True)
        )

        print(len(train_df_base))

        if 'sentence' in train_df_base.columns:
            train_df_base.rename(columns={'sentence': 'text'}, inplace=True)
        train_df_base.drop(columns=[col for col in train_df_base.columns if col not in ['text', 'label']], inplace=True)
    
    train_df_id = train_df_base[train_df_base['label'].isin(id_labels)]
    test_df_id = test_df_base[test_df_base['label'].isin(id_labels)]
    
    for df in [train_df_id, test_df_id]:
        set_binary_label(df, indomain=True)
    
    ood_df_train = train_df_base[train_df_base['label'].isin(ood_labels)]
    ood_df = test_df_base[test_df_base['label'].isin(ood_labels)]
    set_binary_label(ood_df, indomain=False)
    
    return train_df_id, ood_df, test_df_id


def load_datasets_and_create_dataloaders_for_test(df, batch_size = 16, seed=42, separation = False, data_type = 'bert'):

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')   
    
    tokenized_data = tokenizer(df['text'].tolist(), padding='max_length', truncation=True, max_length=512)

    tokenized_data['label'] = df['label'].tolist()

    def create_dataloader(tokenized_dataset, batch_size):
        inputs = torch.tensor(tokenized_dataset['input_ids'])
        masks = torch.tensor(tokenized_dataset['attention_mask'])
        labels = torch.tensor(tokenized_dataset['label'])

        dataset = TensorDataset(inputs, masks, labels)
        generator = torch.Generator()
        generator.manual_seed(42)  # ou ta seed globale
        sampler = RandomSampler(dataset, generator=generator)
        sampler = RandomSampler(dataset)
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)

        return dataloader

    new_dataloader = create_dataloader(tokenized_data, batch_size)

    return new_dataloader

def save_features(id_train, id_test, ood_test, model_name, dataset_name, language):


    base_dir = f'/home/ids/fihey-23/new-code-paper/embeddings_files/{model_name}/{language}/{dataset_name}/' 


    os.makedirs(os.path.dirname(base_dir), exist_ok=True)

    features_map = {
        "train_features": id_train,
        "test_features": id_test,
        "ood_features": ood_test
    }

    for feature_name, features in features_map.items():
        file_name = os.path.join(base_dir, f"{feature_name}.pkl")
        with open(file_name, 'wb') as f:
            pickle.dump(features, f)

def load_features(dataset_name, model_name, language):


    base_dir = f'/home/ids/fihey-23/new-code-paper/embeddings_files/{model_name}/{language}/{dataset_name}/' 

    features_map = {}
    
    for feature_name in ["train_features", "test_features", "ood_features"]:
        file_name = os.path.join(base_dir, f"{feature_name}.pkl")
        
        if os.path.exists(file_name):  # Vérifie que le fichier existe
            with open(file_name, 'rb') as f:
                features_map[feature_name] = pickle.load(f)
                print(f"{feature_name} chargé depuis {file_name}.")
        else:
            raise FileNotFoundError(f"Fichier introuvable : {file_name}")
    
    return features_map

