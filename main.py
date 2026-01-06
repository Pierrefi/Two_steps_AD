#!/usr/bin/env python3
import os
import sys 
import time

# Ajouter le dossier parent au path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from transformers import BertTokenizer, AutoTokenizer, RobertaTokenizer

from torch.nn import CrossEntropyLoss
import torch
from data_processing import create_datasets
from feature_extraction import feature_extract, extract_features_with_flow
 
from datasets import load_dataset
import pandas as pd
from data_processing import save_features, load_features
from detection_algorithms import ood_detection_every_combination
from sklearn.decomposition import PCA
from datasets import concatenate_datasets
import numpy as np

import argparse, time

device = torch.device('cuda:0') 

def load_data(name, lg = None):
    
    if name == 'hume_toxic' :
        dataset = load_dataset("mteb/HUMEToxicConversationsClassification")
        id_labels, ood_labels = [0], [1]

        train_df_id, ood_df, test_df_id = create_datasets(dataset, id_labels, ood_labels, sample = False)

    if name == 'toxic_conversations':
        dataset = load_dataset("mteb/toxic_conversations_50k")

        id_labels, ood_labels = [0], [1]
        train_df_id, ood_df, test_df_id = create_datasets(dataset, id_labels, ood_labels, sample = False)

    if name == 'banking77':
        dataset = load_dataset("mteb/banking77")

        id_labels, ood_labels = list(range(0, 40)), list(range(40, 77))
        train_df_id, ood_df, test_df_id = create_datasets(dataset, id_labels, ood_labels, sample = False)

    if name == 'dbpedia':
        dataset = load_dataset("mteb/DBpediaClassification")

        id_labels, ood_labels = list(range(0, 8)), list(range(8, 13))
        train_df_id, ood_df, test_df_id = create_datasets(dataset, id_labels, ood_labels, sample = False)

    if name == 'yahoo_answers_topic':
        dataset = load_dataset("mteb/YahooAnswersTopicsClassification")

        id_labels, ood_labels = list(range(0, 6)), list(range(6, 9))
        train_df_id, ood_df, test_df_id = create_datasets(dataset, id_labels, ood_labels, sample = False)

    if name == 'arxiv_class': #topic
        dataset = load_dataset("mteb/ArxivClassification")

        id_labels, ood_labels = list(range(0, 7)), list(range(7, 10))
        train_df_id, ood_df, test_df_id = create_datasets(dataset, id_labels, ood_labels, sample = True)

    if name == 'financial_phrasebank': #sent
        dataset = load_dataset("mteb/FinancialPhrasebankClassification")

        id_labels, ood_labels = [2], [0]
        train_df_id, ood_df, test_df_id = create_datasets(dataset, id_labels, ood_labels, sample = False)

    if name == 'tweet_sentiment': #sent
        dataset = load_dataset("mteb/tweet_sentiment_extraction")

        id_labels, ood_labels = [2], [0]
        train_df_id, ood_df, test_df_id = create_datasets(dataset, id_labels, ood_labels, sample = True)

    if name == 'patent_class': #topic
        dataset = load_dataset("mteb/PatentClassification")

        id_labels, ood_labels = list(range(0, 5)), list(range(5, 8))
        train_df_id, ood_df, test_df_id = create_datasets(dataset, id_labels, ood_labels, sample = True)

    if name == 'yelp_reviews': #sent
        dataset = load_dataset("Yelp/yelp_review_full")

        id_labels, ood_labels = [4], [0]
        train_df_id, ood_df, test_df_id = create_datasets(dataset, id_labels, ood_labels, sample = True)

    ###################### ML ########################

    if name == 'multi_hate_classification': #ara, deu, eng, fra, ita, nld, hin, pol, por, spa
        dataset = load_dataset("mteb/multi_hate", lg)

        id_labels, ood_labels = [0], [1]
        train_df_id, ood_df, test_df_id = create_datasets(dataset, id_labels, ood_labels, sample = False)  

    if name == 'massive_intent': # ar, de, en, fr, it, nl, hi, pl, pt, es, th
        dataset = load_dataset("mteb/MassiveIntentClassification", lg)

        id_labels, ood_labels = list(range(0, 40)), list(range(40, 60))
        train_df_id, ood_df, test_df_id = create_datasets(dataset, id_labels, ood_labels, sample = False) 

    if name == 'massive_scenario': # ar, de, en, fr, it, nl, hi, pl, pt, es, th
        dataset = load_dataset("mteb/massive_scenario", lg)

        id_labels, ood_labels = list(range(0, 10)), list(range(10, 18))
        train_df_id, ood_df, test_df_id = create_datasets(dataset, id_labels, ood_labels, sample = False)     

    if name == 'mtop_intent': #de, es, fr, en, hi, th
        dataset = load_dataset("mteb/MTOPIntentClassification", lg)

        id_labels, ood_labels = list(range(0, 80)), list(range(80, 111))
        train_df_id, ood_df, test_df_id = create_datasets(dataset, id_labels, ood_labels, sample = False)    

    if name == 'mtop_domain': #de, es, fr, en, hi, th
        dataset = load_dataset("mteb/MTOPDomainClassification", lg)

        id_labels, ood_labels = list(range(0, 7)), list(range(7, 10))
        train_df_id, ood_df, test_df_id = create_datasets(dataset, id_labels, ood_labels, sample = False)   

    if name == 'mashaka_news': #eng, fra, amh, hau, ibo, lin, lug, orm, pcm, run, swa, sna, som, tir, who, yor 
        dataset = load_dataset("mteb/masakhanews", lg)

        id_labels, ood_labels = list(range(0, 5)), list(range(5, 7))
        train_df_id, ood_df, test_df_id = create_datasets(dataset, id_labels, ood_labels, sample = False)        

    if name == 'amazon_reviews': #de, en, es, fr, ja, zh
        dataset = load_dataset("mteb/AmazonReviewsClassification", lg)

        id_labels, ood_labels = [3], [0]
        train_df_id, ood_df, test_df_id = create_datasets(dataset, id_labels, ood_labels, sample = True) 

    if name == 'amazon_counterfactual': #de, en, ja
        dataset = load_dataset("mteb/amazon_counterfactual", lg)

        id_labels, ood_labels = [1], [0]
        train_df_id, ood_df, test_df_id = create_datasets(dataset, id_labels, ood_labels, sample = False)    


    ########################## German ############################

    if name == 'ten_kgnad_topic':
        dataset = load_dataset("mteb/TenKGnadClassification")

        id_labels, ood_labels = list(range(0, 5)), list(range(5, 8))
        train_df_id, ood_df, test_df_id = create_datasets(dataset, id_labels, ood_labels, sample = False)   

    ########################## Spanish ############################

    if name == 'spanish_news':
        dataset = load_dataset("mteb/SpanishNewsClassification")

        id_labels, ood_labels = list(range(0, 7)), list(range(7, 10))
        train_df_id, ood_df, test_df_id = create_datasets(dataset, id_labels, ood_labels, sample = False)   
  
    if name == 'spanish_sentiment': #sent
        dataset = load_dataset("sepidmnorozy/Spanish_sentiment")

        id_labels, ood_labels = [1], [0]
        train_df_id, ood_df, test_df_id = create_datasets(dataset, id_labels, ood_labels, sample = False)   
  
    ########################## Italian ############################

    if name == 'ita_casehold':
        dataset = load_dataset("mteb/ItaCaseholdClassification")

        id_labels, ood_labels = list(range(0, 50)), list(range(50, 71))
        train_df_id, ood_df, test_df_id = create_datasets(dataset, id_labels, ood_labels, sample = False)   
  
    if name == 'sardistance': #sent
        dataset = load_dataset("MattiaSangermano/SardiStance")

        id_labels, ood_labels = [1], [0]
        train_df_id, ood_df, test_df_id = create_datasets(dataset, id_labels, ood_labels, sample = False)   
  
    ########################## Portuguese ############################

    if name == 'hate_speech_portuguese':
        dataset = load_dataset("mteb/HateSpeechPortugueseClassification")

        id_labels, ood_labels = [0], [1]
        train_df_id, ood_df, test_df_id = create_datasets(dataset, id_labels, ood_labels, sample = False)   

    ########################## French ############################

    if name == 'movie_reviews':
        dataset = load_dataset("mteb/MovieReviewSentimentClassification")

        id_labels, ood_labels = [1], [0]
        train_df_id, ood_df, test_df_id = create_datasets(dataset, id_labels, ood_labels, sample = True)  

    ########################## Thai ############################

    if name == 'wongnai': #sent
        dataset = load_dataset("Wongnai/wongnai_reviews")

        def rename_columns(example):
            return {
                "text": example["review_body"],
                "label": example["star_rating"]
            }

        dataset = dataset.map(rename_columns,remove_columns=[c for c in dataset["train"].column_names if c not in ("review_body", "star_rating")])
        id_labels, ood_labels = [4,5], [1]
        train_df_id, ood_df, test_df_id = create_datasets(dataset, id_labels, ood_labels, sample = False)  

    if name == 'wisesight': #sent
        dataset = load_dataset("mteb/WisesightSentimentClassification")

        id_labels, ood_labels = [0], [2]
        train_df_id, ood_df, test_df_id = create_datasets(dataset, id_labels, ood_labels, sample = False)  

    ########################## Hindi ############################

    if name == 'sent_hindi': #sent
        dataset = load_dataset("mteb/sentiment_analysis_hindi")

        id_labels, ood_labels = [1], [2]
        train_df_id, ood_df, test_df_id = create_datasets(dataset, id_labels, ood_labels, sample = False)  

    return train_df_id, ood_df, test_df_id




def task_extract_features(dataset_names, contamination, model_name, data_tag, device, language, lg):
    for dataset_name in dataset_names:
        # lg = code HF (ex: fr/en/de/it...), language = tag expérimental (ex: french/english/...)
        train_df_id, ood_df, test_df_id = load_data(dataset_name, lg=lg)

        features_id_train, features_id_test, features_ood = feature_extract(
            train_df_id, test_df_id, ood_df, device, model_name
        )

        save_features(features_id_train, features_id_test, features_ood,
                      model_name, dataset_name, language)

        print(f"[OK] Features sauvées → model={model_name} dataset={dataset_name} data_tag={data_tag} "
              f"language={language} lg={lg}")


def task_do_test(dataset_names, model_name, data_tag, methods, seed, language, post_proc_method=''):
    """
    Charge les features sauvegardées et lance les tests OOD pour chaque méthode.
    """
    rng = np.random.default_rng(seed)

    for dataset_name in dataset_names:

        print('Dataset : ', dataset_name)
        features = load_features(dataset_name, model_name, language=language)

        train_features = features["train_features"]
        test_features = features["test_features"]
        ood_features = features["ood_features"]

        print(f"Train features size: {train_features.shape[0]}")
        print(f"Validation features size: {test_features.shape[0]}")
        print(f"OOD features size: {ood_features.shape[0]}")

        for method in methods:
            start_time = time.perf_counter()
            if post_proc_method == '':
                ood_detection_every_combination(
                    '',
                    train_features, test_features, ood_features,
                    'no_post_processing',
                    method, dataset_name, model_name,
                    language=language
                )
            else:
                ood_detection_every_combination(
                    '',
                    train_features, test_features, ood_features,
                    post_proc_method,
                    method, dataset_name, model_name,
                    language=language
                )
            end_time = time.perf_counter()
            duration = end_time - start_time
            print(f"⏱️ Temps écoulé : {duration/60:.2f} min ({duration:.1f} s)")


def parse_list(arg: str):
    """Transforme 'a,b,c' -> ['a','b','c'] en ignorant espaces/vides."""
    if arg is None or arg == "":
        return []
    return [x.strip() for x in arg.split(",") if x.strip()]


def pick_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    p = argparse.ArgumentParser(description="Runner simple pour extract_features / do_test (arguments minimalistes).")
    p.add_argument("--task", required=True, choices=["extract_features", "do_test"],
                   help="Tâche à exécuter.")
    p.add_argument("--dataset_names", required=True,
                   help="Liste de datasets séparés par des virgules (ex: 'banking,clinc').")
    p.add_argument("--contamination", type=float, default=0.0,
                   help="Taux de contamination pour dataloaders_creation (uniquement pour extract_features).")
    p.add_argument("--model_name", required=True,
                   help="Nom du modèle pour sauvegarde/chargement des features (ex: 'sbert_cont_40').")
    p.add_argument("--methods", default="",
                   help="Liste de méthodes séparées par des virgules pour do_test (ex: 'knn,ocsvm,projection_depth').")
    p.add_argument("--post_proc", default="")
    p.add_argument("--language", default="english",
                   help="Tag expérimental (ex: english/french/german). Sert à organiser les sorties.")
    p.add_argument("--lg", default=None,
                   help="Code langue HuggingFace (ex: en, fr, de, it). Utilisé pour load_dataset(dataset, lg).")
    p.add_argument("--seed", type=int, default=42, help="Graine aléatoire pour l'échantillonnage.")
    p.add_argument("--data_tag", default="agnews",
                   help="Tag 'data' passé à save_features / load_features (défaut: agnews).")

    args = p.parse_args()

    dataset_names = parse_list(args.dataset_names)
    methods = parse_list(args.methods)

    device = pick_device()
    print(f"[info] device={device}")

    if args.task == "extract_features":
        task_extract_features(
            dataset_names=dataset_names,
            contamination=args.contamination,
            model_name=args.model_name,
            data_tag=args.data_tag,
            device=device,
            language=args.language,
            lg=args.lg
        )
    elif args.task == "do_test":
        if not methods:
            methods = ['autoencoder', 'gmm', 'lof', 'isolation_forest', 'ocsvm',
                       'knn', 'lunar']
        task_do_test(
            dataset_names=dataset_names,
            model_name=args.model_name,
            data_tag=args.data_tag,
            methods=methods,
            seed=args.seed,
            language=args.language,
            post_proc_method=args.post_proc
        )


if __name__ == "__main__":
    main()
