#!/usr/bin/env python3

from __future__ import absolute_import, division, print_function

import sys, os
sys.path.append("/home/ids/fihey-23/new_project/pytorch-bertflow")
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import MinCovDet
import numpy as np 
import pandas as pd 
from tqdm import tqdm
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModel, AutoConfig, AutoModelForCausalLM
import torch.nn.functional as F

# --- BERT-Flow (depuis le repo bohanli/BERT-flow) ---
from tflow_utils import TransformerGlow, AdamWeightDecayOptimizer

device = torch.device('cuda:0') 

def mean_pooling(last_hidden_state, attention_mask):

    input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size())    
    sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, dim=1)    
    sum_mask = input_mask_expanded.sum(dim=1).clamp(min=1e-9)
    
    return sum_embeddings / sum_mask

def compute_whitening_from_dataloader(model, dataloader, device):
    model.eval()
    all_embeddings = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids, attention_mask, _ = (t.to(device) for t in batch)

            embeddings = model.aggregated_layers_features(input_ids=input_ids,attention_mask=attention_mask,aggregation_method="mean")
            all_embeddings.append(embeddings.cpu())

    X = torch.cat(all_embeddings, dim=0).numpy()  # (n_samples, hidden_size)

    mu = X.mean(0)
    X_centered = X - mu
    cov = np.cov(X_centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    W = eigvecs @ np.diag(1.0 / np.sqrt(eigvals + 1e-5)) @ eigvecs.T

    return mu, W

def feature_extract(train_df_id, test_df_id, ood_df, device, model_type): 
        
    if model_type == 'xlm-roberta-base':
        tokenizer = AutoTokenizer.from_pretrained("FacebookAI/xlm-roberta-base")
        model = AutoModel.from_pretrained("FacebookAI/xlm-roberta-base").to(device)

    if model_type == 'sbert_ml':
        model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2').to(device)

        def extract_features(texts, model, batch_size=8):
            return model.encode(texts,batch_size=batch_size,convert_to_tensor=True, show_progress_bar=True,device=device).cpu().numpy()

        features_id_train = extract_features(train_df_id['text'].tolist(), model)
        features_id_val = extract_features(test_df_id['text'].tolist(), model)
        features_ood = extract_features(ood_df['text'].tolist(), model)

        return features_id_train, features_id_val, features_ood

    if model_type == 'qwen3':
        model = SentenceTransformer('Qwen/Qwen3-Embedding-0.6B', trust_remote_code=True).to(device)
 
        def extract_features(texts, model, batch_size=2):
            return model.encode(texts,batch_size=batch_size,convert_to_tensor=True, show_progress_bar=True,device=device).cpu().numpy()

        features_id_train = extract_features(train_df_id['text'].tolist(), model)
        features_id_val = extract_features(test_df_id['text'].tolist(), model)
        features_ood = extract_features(ood_df['text'].tolist(), model)

        return features_id_train, features_id_val, features_ood
        
    if model_type == 'e5':
        model = SentenceTransformer("intfloat/multilingual-e5-base").to(device)
    
        def extract_features(texts, model, batch_size=8):
            return model.encode(texts,batch_size=batch_size,convert_to_tensor=True ,show_progress_bar=True,device=device).cpu().numpy()

        features_id_train = extract_features(train_df_id['text'].tolist(), model)
        features_id_val = extract_features(test_df_id['text'].tolist(), model)
        features_ood = extract_features(ood_df['text'].tolist(), model)

        return features_id_train, features_id_val, features_ood

    if model_type == 'llama':

        tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.2-1B')
        model = AutoModel.from_pretrained('meta-llama/Llama-3.2-1B').to(device)
    
    if model_type == "qwen3llm":

        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B").to(device)

        def _sanitize_batch(batch_texts, tokenizer):
            clean = []
            for t in batch_texts:
                if t is None:
                    t = ""
                if not isinstance(t, str):
                    t = str(t)
                t = t.strip()
                if t == "":
                    t = tokenizer.eos_token if tokenizer.eos_token is not None else " "
                clean.append(t)
            return clean

        def extract_features(df, batch_size=4):
            texts = df["text"].tolist()
            embeddings = []

            model.eval()
            with torch.no_grad():
                for i in tqdm(range(0, len(texts), batch_size)):
                    batch_texts = _sanitize_batch(texts[i:i + batch_size], tokenizer)

                    tokens = tokenizer(
                        batch_texts,
                        padding=True,
                        return_tensors="pt",
                        truncation=True,
                        max_length=1024,
                        add_special_tokens=True,
                    )
                    tokens = {k: v.to(device) for k, v in tokens.items()}

                    # sécurité ultime: éviter seq_len=0
                    if tokens["input_ids"].shape[1] == 0:
                        eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
                        tokens["input_ids"] = torch.tensor([[eos_id]], device=device)
                        tokens["attention_mask"] = torch.tensor([[1]], device=device)

                    outputs = model(**tokens, output_hidden_states=True)
                    last_hidden = outputs.hidden_states[-1]  # (B, L, D)

                    pooled = mean_pooling(last_hidden, tokens["attention_mask"])
                    pooled = F.normalize(pooled, p=2, dim=1)
                    embeddings.append(pooled.cpu())

            return torch.cat(embeddings, dim=0).numpy()

        features_id_train = extract_features(train_df_id)
        features_id_val   = extract_features(test_df_id)
        features_ood      = extract_features(ood_df)

        return features_id_train, features_id_val, features_ood

    def extract_features_from_df(df):
        texts = df['text'].tolist()
        batch_size = 8
        embeddings = []
        model.eval()

        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))

        with torch.no_grad():
            for i in tqdm(range(0, len(texts), batch_size)):
                batch_texts = texts[i:i+batch_size]
                tokens = tokenizer(batch_texts, padding=True, return_tensors="pt", truncation = True, max_length = 512).to(device)
                outputs = model(**tokens)
                pooled = mean_pooling(outputs.last_hidden_state, tokens['attention_mask'])
                pooled = F.normalize(pooled, p=2, dim=1)
                embeddings.append(pooled)

        return torch.cat(embeddings, dim=0).cpu().numpy()

    features_id_train = extract_features_from_df(train_df_id)
    features_id_val = extract_features_from_df(test_df_id)
    features_id_ood = extract_features_from_df(ood_df)

    return features_id_train, features_id_val, features_id_ood


def extract_features_with_flow(train_df_id, test_df_id, ood_df, device, model_type):

    # --- Choix du backbone + pooling en fonction du model_type ---
    if model_type == "sbertflow":
        backbone_name = "sentence-transformers/all-mpnet-base-v2"
        pooling = "first-last-avg"   # comme dans ton code BERTFlow
        bf_max_len = 1024

    elif model_type == "llamaflow":
        backbone_name = "meta-llama/Llama-3.2-1B"
        pooling = "mean"             # équivalent de ton mean_pooling + normalisation
        bf_max_len = 1024            # tu utilisais 1024 pour les LLMs

    elif model_type == "gemmaflow":
        backbone_name = "google/gemma-3-270m"
        pooling = "mean"             # idem, on fait un pooling moyen
        bf_max_len = 1024

    else:
        raise ValueError(f"with_flow() ne gère pas ce model_type : {model_type}")

    # --- Tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(backbone_name, trust_remote_code=True)

    # Gestion du pad_token pour les modèles causal LM (LLaMA / Gemma)
    if tokenizer.pad_token is None:
        if hasattr(tokenizer, "eos_token") and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            # fallback au cas où
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    # --- Modèle Flow au-dessus du backbone HF ---
    flow_model = TransformerGlow(backbone_name, pooling=pooling).to(device)
    flow_model.train()  # on n'entraîne que le flow (flow_model.glow)

    # --- Optimiseur : uniquement les paramètres du flow ---
    no_decay = ["bias", "LayerNorm.weight"]
    glow_params = list(flow_model.glow.named_parameters())
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in glow_params if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.01,
        },
        {
            "params": [p for n, p in glow_params if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamWeightDecayOptimizer(
        params=optimizer_grouped_parameters,
        lr=1e-3,
        eps=1e-6,
    )

    # --- Hyperparams Flow ---
    bf_epochs     = 30
    bf_batch_size = 16
    texts = train_df_id['text'].tolist()

    # --- Entraînement du Flow ---
    for epoch in range(bf_epochs):
        np.random.shuffle(texts)  # très important pour entraîner le flow
        total_loss = 0.0

        for i in tqdm(
            range(0, len(texts), bf_batch_size),
            desc=f"{model_type}-Flow epoch {epoch+1}/{bf_epochs}"
        ):
            batch = texts[i:i+bf_batch_size]
            enc = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=bf_max_len,
            ).to(device)

            # z: embeddings dans l’espace latent gaussien
            _, loss = flow_model(enc["input_ids"], enc["attention_mask"], return_loss=True)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(flow_model.glow.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

        print(f"→ {model_type}-Flow Epoch {epoch+1}: mean loss = {total_loss/len(texts):.4f}")

    flow_model.eval()

    # --- Encodage avec le Flow (z ~ N(0, I)) ---
    @torch.no_grad()
    def encode_with_flow(sentences):
        embs = []
        for i in range(0, len(sentences), bf_batch_size):
            batch = sentences[i:i+bf_batch_size]
            enc = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=bf_max_len,
            ).to(device)
            z = flow_model(enc["input_ids"], enc["attention_mask"], return_loss=False)
            if isinstance(z, tuple):
                z = z[0]
            embs.append(z.detach().cpu().numpy())
        return np.vstack(embs)

    # --- Features finales pour ID train / val / OOD ---
    features_id_train = encode_with_flow(train_df_id['text'].tolist())
    features_id_val   = encode_with_flow(test_df_id['text'].tolist())
    features_ood      = encode_with_flow(ood_df['text'].tolist())

    return features_id_train, features_id_val, features_ood