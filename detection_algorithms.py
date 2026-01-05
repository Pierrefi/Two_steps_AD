#!/usr/bin/env python3

from __future__ import absolute_import, division, print_function

import os 
from scipy.signal import savgol_filter, argrelextrema
from umap import UMAP 
from sklearn.preprocessing import normalize
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import MinCovDet
import hdbscan
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import kneighbors_graph
from scipy.sparse.csgraph import shortest_path
from sklearn.metrics import pairwise_distances
import numpy as np 
from scipy import stats
from sklearn.metrics import roc_auc_score
from tqdm import tqdm 

from pyod.models.lunar import LUNAR
from pyod.models.auto_encoder import AutoEncoder
from pyod.models.lof import LOF
from pyod.models.gmm import GMM
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.ocsvm import OCSVM

import torch
import torch.nn.functional as F
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
#from depth.multivariate import *
from depth.model import DepthEucl

import plotly.express as px

from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import check_random_state
from sklearn.neighbors import NearestNeighbors
from scipy.stats import pearsonr
import calculate_log as callog
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest

device = torch.device('cuda:0') 
random_state = 53

def detection_performance(scores, Y, outf, tag='TMP'):
    import os 
    os.makedirs(outf, exist_ok=True)
    num_samples = scores.shape[0]
    l1 = open('%s/confidence_%s_In.txt'%(outf, tag), 'w')
    l2 = open('%s/confidence_%s_Out.txt'%(outf, tag), 'w')
    y_pred = scores 

    for i in range(num_samples):
        if Y[i] == 0:
            l1.write("{}\n".format(-y_pred[i]))
        else:
            l2.write("{}\n".format(-y_pred[i]))
    l1.close()
    l2.close()
    results = callog.metric(outf, [tag])
    return results

def append_to_latex_table(file_path, model_name, results, mtypes=None):
    if mtypes is None:
        mtypes = ['AUROC', 'DTACC', 'AUIN', 'AUOUT']
    print(os)
    dir_name = os.path.dirname(file_path)
    if dir_name and not os.path.exists(dir_name):
        os.makedirs(dir_name)

    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        # Remove the last few lines (\hline and \end{tabular}) so we can append new data
        lines = lines[:-3]

        # Append new row to the LaTeX table
        with open(file_path, 'w') as f:
            f.writelines(lines)  # Write back all the lines except \hline and \end{tabular}
            f.write(f'\\textbf{{{model_name}}} & ' + ' & '.join([f'{100.*results["TMP"][mtype]:.2f}' for mtype in mtypes]) + ' \\\\\n')
            f.write('\\hline\n')
            f.write('    \\end{tabular}\n')
            f.write('\\caption{Metrics Results}\n')
            f.write('\\label{tab:metrics}\n')
            f.write('\\end{table}\n')
    else:
        # Create a new LaTeX table with headers
        with open(file_path, 'w') as f:
            f.write("\\begin{table}[h]\n")
            f.write("    \\centering\n")
            f.write("    \\begin{tabular}{|c|" + "c|"*len(mtypes) + "}\n")
            f.write("        \\hline\n")
            f.write("         Model & " + ' & '.join([f'\\textbf{{{mtype}}}' for mtype in mtypes]) + " \\\\\n")
            f.write("         \\hline\n")
            f.write(f'\\textbf{{{model_name}}} & ' + ' & '.join([f'{100.*results["TMP"][mtype]:.2f}' for mtype in mtypes]) + ' \\\\\n')
            f.write('        \\hline\n')
            f.write('    \\end{tabular}\n')
            f.write('\\caption{Metrics Results}\n')
            f.write('\\label{tab:metrics}\n')
            f.write('\\end{table}\n')

    print(f"LaTeX table updated and saved to {file_path}")

def whitening(X):

    mu = np.mean(X, axis=0)
    X_centered = X - mu
    cov = np.cov(X_centered, rowvar=False)
    values, vectors = np.linalg.eigh(cov)
    W = vectors @ np.diag(1.0 / np.sqrt(values + 1e-5)) @ vectors.T

    return X_centered @ W

def compute_whitening_params(X):

    mu = np.mean(X, axis=0)
    X_centered = X - mu
    cov = np.cov(X_centered, rowvar=False)
    values, vectors = np.linalg.eigh(cov)
    W = vectors @ np.diag(1.0 / np.sqrt(values + 1e-5)) @ vectors.T  
    return mu, W


def apply_whitening(X, mu, W):

    X_centered = X - mu
    return X_centered @ W


def ood_detection_every_combination(loss_fn_name, features_id_train, features_id_val, features_ood, post_proc_method, method, dataset_name, model_type, language, support_fraction = 0.6): 

  
    if post_proc_method == 'train_and_test_whitening_with_train_params':

        mu_train, W_train = compute_whitening_params(features_id_train)
        features_id_train = apply_whitening(features_id_train, mu_train, W_train)

        features_combined = np.concatenate((features_id_val, features_ood), axis=0)
        features_combined = apply_whitening(features_combined, mu_train, W_train)
        num_id_val = features_id_val.shape[0]
        features_id_val = features_combined[:num_id_val]
        features_ood = features_combined[num_id_val:num_id_val + features_ood.shape[0]]

        loss_fn_name += '_with_train_and_test_whitening_with_train_params'

    if method == 'ocsvm' : 

        ocsvm_model = OCSVM()  
        ocsvm_model.fit(features_id_train)

        test_scores = ocsvm_model.decision_function(features_id_val)  
        ood_scores = ocsvm_model.decision_function(features_ood)

        X_scores = np.concatenate((ood_scores, test_scores))
        ood_labels = np.ones_like(ood_scores)
        test_labels = np.zeros_like(test_scores)
        Y_test = np.concatenate((ood_labels, test_labels))

        results = detection_performance(X_scores, Y_test, 'feats_logs', tag='TMP')

        file_path = f'/home/ids/fihey-23/new-code-paper/results_tables/{language}/{dataset_name}_results/{model_type}/{model_type}_ocsvm.tex'
  
        append_to_latex_table(file_path, loss_fn_name, results)

        mtypes = ['AUROC', 'DTACC', 'AUIN', 'AUOUT']
        for mtype in mtypes:
            print(' {mtype:6s}'.format(mtype=mtype), end='')
        print('\n{val:6.2f}'.format(val=100.*results['TMP']['AUROC']), end='')
        print(' {val:6.2f}'.format(val=100.*results['TMP']['DTACC']), end='')
        print(' {val:6.2f}'.format(val=100.*results['TMP']['AUIN']), end='')
        print(' {val:6.2f}\n'.format(val=100.*results['TMP']['AUOUT']), end='')
    
    if method == 'lof' : 

        lof_model = LOF()  
        lof_model.fit(features_id_train)

        test_scores = lof_model.decision_function(features_id_val)  
        ood_scores = lof_model.decision_function(features_ood)

        X_scores = np.concatenate((ood_scores, test_scores))
        ood_labels = np.ones_like(ood_scores)
        test_labels = np.zeros_like(test_scores)
        Y_test = np.concatenate((ood_labels, test_labels))

        results = detection_performance(X_scores, Y_test, 'feats_logs', tag='TMP')
        file_path = f'/home/ids/fihey-23/new-code-paper/results_tables/{language}/{dataset_name}_results/{model_type}/{model_type}_lof.tex'
 
        append_to_latex_table(file_path, loss_fn_name, results)

        mtypes = ['AUROC', 'DTACC', 'AUIN', 'AUOUT']
        for mtype in mtypes:
            print(' {mtype:6s}'.format(mtype=mtype), end='')
        print('\n{val:6.2f}'.format(val=100.*results['TMP']['AUROC']), end='')
        print(' {val:6.2f}'.format(val=100.*results['TMP']['DTACC']), end='')
        print(' {val:6.2f}'.format(val=100.*results['TMP']['AUIN']), end='')
        print(' {val:6.2f}\n'.format(val=100.*results['TMP']['AUOUT']), end='')
    
    if method == 'isolation_forest':

        if_model = IForest()  
        if_model.fit(features_id_train)

        test_scores = if_model.decision_function(features_id_val)  
        ood_scores = if_model.decision_function(features_ood)

        X_scores = np.concatenate((ood_scores, test_scores))
        ood_labels = np.ones_like(ood_scores)
        test_labels = np.zeros_like(test_scores)
        Y_test = np.concatenate((ood_labels, test_labels))

        results = detection_performance(X_scores, Y_test, 'feats_logs', tag='TMP')
        file_path = f'/home/ids/fihey-23/new-code-paper/results_tables/{language}/{dataset_name}_results/{model_type}/{model_type}_isolation_forest.tex'

        append_to_latex_table(file_path, loss_fn_name, results)

        mtypes = ['AUROC', 'DTACC', 'AUIN', 'AUOUT']
        for mtype in mtypes:
            print(' {mtype:6s}'.format(mtype=mtype), end='')
        print('\n{val:6.2f}'.format(val=100.*results['TMP']['AUROC']), end='')
        print(' {val:6.2f}'.format(val=100.*results['TMP']['DTACC']), end='')
        print(' {val:6.2f}'.format(val=100.*results['TMP']['AUIN']), end='')
        print(' {val:6.2f}\n'.format(val=100.*results['TMP']['AUOUT']), end='')

    if method == 'knn': 

        knn_model = KNN()  
        knn_model.fit(features_id_train)

        # Get anomaly scores for validation and OOD datasets
        test_scores = knn_model.decision_function(features_id_val)  
        ood_scores = knn_model.decision_function(features_ood)

        # Combine scores and labels
        X_scores = np.concatenate((ood_scores, test_scores))
        ood_labels = np.ones_like(ood_scores)
        test_labels = np.zeros_like(test_scores)
        Y_test = np.concatenate((ood_labels, test_labels))

        # Evaluate detection performance
        results = detection_performance(X_scores, Y_test, 'feats_logs', tag='TMP')

        file_path = f'/home/ids/fihey-23/new-code-paper/results_tables/{language}/{dataset_name}_results/{model_type}/{model_type}_knn.tex'

        append_to_latex_table(file_path, loss_fn_name, results)

        # Affichage des meilleurs résultats
        mtypes = ['AUROC', 'DTACC', 'AUIN', 'AUOUT']
        for mtype in mtypes:
            print(' {mtype:6s}'.format(mtype=mtype), end='')
        print('\n{val:6.2f}'.format(val=100.*results['TMP']['AUROC']), end='')
        print(' {val:6.2f}'.format(val=100.*results['TMP']['DTACC']), end='')
        print(' {val:6.2f}'.format(val=100.*results['TMP']['AUIN']), end='')
        print(' {val:6.2f}\n'.format(val=100.*results['TMP']['AUOUT']), end='')

    if method == 'gmm': 

        gmm_model = GMM()  
        gmm_model.fit(features_id_train)

        # Get anomaly scores for validation and OOD datasets
        test_scores = gmm_model.decision_function(features_id_val)  
        ood_scores = gmm_model.decision_function(features_ood)

        # Combine scores and labels
        X_scores = np.concatenate((ood_scores, test_scores))
        ood_labels = np.ones_like(ood_scores)
        test_labels = np.zeros_like(test_scores)
        Y_test = np.concatenate((ood_labels, test_labels))

        # Evaluate detection performance
        results = detection_performance(X_scores, Y_test, 'feats_logs', tag='TMP')
        
        file_path = f'/home/ids/fihey-23/new-code-paper/results_tables/{language}/{dataset_name}_results/{model_type}/{model_type}_gmm.tex'

        append_to_latex_table(file_path, loss_fn_name, results)
        mtypes = ['AUROC', 'DTACC', 'AUIN', 'AUOUT']
        for mtype in mtypes:
            print(f' {mtype:6s}', end='')
        print(f'\n{results["TMP"]["AUROC"] * 100:6.2f}', end='')
        print(f' {results["TMP"]["DTACC"] * 100:6.2f}', end='')
        print(f' {results["TMP"]["AUIN"] * 100:6.2f}', end='')
        print(f' {results["TMP"]["AUOUT"] * 100:6.2f}\n', end='')

    if method == 'lunar':

        lunar_model = LUNAR()  
        lunar_model.fit(features_id_train)

        # Get anomaly scores for validation and OOD datasets
        test_scores = lunar_model.decision_function(features_id_val)  # Higher scores mean more abnormal
        ood_scores = lunar_model.decision_function(features_ood)

        # Combine scores and labels
        X_scores = np.concatenate((ood_scores, test_scores))
        ood_labels = np.ones_like(ood_scores)
        test_labels = np.zeros_like(test_scores)
        Y_test = np.concatenate((ood_labels, test_labels))

        # Evaluate detection performance
        results = detection_performance(X_scores, Y_test, 'feats_logs', tag='TMP')
        
        file_path = f'/home/ids/fihey-23/new-code-paper/results_tables/{language}/{dataset_name}_results/{model_type}/{model_type}_lunar.tex'

        append_to_latex_table(file_path, loss_fn_name, results)

        # Print the best metrics
        mtypes = ['AUROC', 'DTACC', 'AUIN', 'AUOUT']
        for mtype in mtypes:
            print(' {mtype:6s}'.format(mtype=mtype), end='')
        print('\n{val:6.2f}'.format(val=100.*results['TMP']['AUROC']), end='')
        print(' {val:6.2f}'.format(val=100.*results['TMP']['DTACC']), end='')
        print(' {val:6.2f}'.format(val=100.*results['TMP']['AUIN']), end='')
        print(' {val:6.2f}\n'.format(val=100.*results['TMP']['AUOUT']), end='')

    if method == 'autoencoder':
            
        autoencoder = AutoEncoder()        
        autoencoder.fit(features_id_train)

        # Obtenir les scores d'anomalie (erreur de reconstruction)
        test_scores = autoencoder.decision_function(features_id_val)
        ood_scores = autoencoder.decision_function(features_ood)

        # Combinaison des scores et des étiquettes
        X_scores = np.concatenate((ood_scores, test_scores))
        ood_labels = np.ones_like(ood_scores)
        test_labels = np.zeros_like(test_scores)
        Y_test = np.concatenate((ood_labels, test_labels))

        # Évaluation des performances de détection
        results = detection_performance(X_scores, Y_test, 'feats_logs', tag='TMP')
        

        # Comparaison entre scores normaux et inversés

        file_path = f'/home/ids/fihey-23/new-code-paper/results_tables/{language}/{dataset_name}_results/{model_type}/{model_type}_autoencoder.tex'

        append_to_latex_table(file_path, loss_fn_name, results)

        # Affichage des meilleurs résultats
        mtypes = ['AUROC', 'DTACC', 'AUIN', 'AUOUT']
        for mtype in mtypes:
            print(' {mtype:6s}'.format(mtype=mtype), end='')
        print('\n{val:6.2f}'.format(val=100.*results['TMP']['AUROC']), end='')
        print(' {val:6.2f}'.format(val=100.*results['TMP']['DTACC']), end='')
        print(' {val:6.2f}'.format(val=100.*results['TMP']['AUIN']), end='')
        print(' {val:6.2f}\n'.format(val=100.*results['TMP']['AUOUT']), end='')


