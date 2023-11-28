# %%
"""
==============================================================
Attempt to classify MEG data in the source space - PLV from matlab code - classification on longer trials w/ SVM

===============================================================

"""
# Authors: Marie-Constance Corsi <marie.constance.corsi@gmail.com>
#
# License: BSD (3-clause)


import os.path as osp
import os

import scipy.stats
import seaborn as sns
import matplotlib.pyplot as plt

import pandas as pd
import mat73

from tqdm import tqdm
import gzip
import pickle
import mne
from mne.connectivity import spectral_connectivity
from mne import create_info, EpochsArray
from mne.decoding import CSP as CSP_MNE
from mne.connectivity import spectral_connectivity

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import ShuffleSplit, cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score

import numpy as np
from scipy.stats import zscore
from moabb.paradigms import MotorImagery

# to compute PLV estimations for each trial
from Scripts.py_viz.fc_pipeline import FunctionalTransformer

# %%
if os.path.basename(os.getcwd()) == "Fenicotteri-equilibristi":
    os.chdir("Database/1_Clinical/Epilepsy_GMD/")
if os.path.basename(os.getcwd()) == "py_viz":
    os.chdir("/Users/marieconstance.corsi/Documents/GitHub/Fenicotteri-equilibristi/Database/1_Clinical/Epilepsy_GMD")
basedir = os.getcwd()

path_csv_root = os.getcwd() + '/1_Dataset-csv/'
if not osp.exists(path_csv_root):
    os.mkdir(path_csv_root)
path_data_root = os.getcwd()
if not osp.exists(path_data_root):
    os.mkdir(path_data_root)
path_data_root_chan = os.getcwd()


# %% functions

def _build_df_plv_from_scores(scores, ppl_name, nbSplit, fmin, fmax):
    pd_PLV_classif_res = pd.DataFrame.from_dict(scores)
    ppl_plv = [ppl_name] * len(pd_PLV_classif_res)
    pd_PLV_classif_res["pipeline"] = ppl_plv
    pd_PLV_classif_res["split"] = [nbSplit] * len(ppl_plv)
    pd_PLV_classif_res["freq"] = [str(fmin) + '-' + str(fmax)] * len(ppl_plv)

    return  pd_PLV_classif_res

#%% parameters for classification
# kk_components = 8
nbSplit = 50
scoring = ['precision', 'recall', 'accuracy', 'f1', 'roc_auc']
# recall = tp / (tp + fn) # default avg binary, to be changed if unbalanced/more classes - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html#sklearn.metrics.recall_score
# precision = tp / (tp + fp) # default avg binary, to be changed if unbalanced/more classes - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html#sklearn.metrics.precision_score
# f1 = F1 = 2 * (precision * recall) / (precision + recall) # here binary & balanced, otherwise need to be changed - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score
# roc_auc - Area under the receiver operating characteristic curve from prediction scores, default macro, to be changed for unbalanced and more classes - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html#sklearn.metrics.roc_auc_score

# shuffle order of epochs & labels
rng = np.random.default_rng(42)  # set a random seed
n_perms = 1#00  # number of permutations wanted

# parameters for the default classifier & cross-validation
svm = GridSearchCV(SVC(), {"kernel": ("linear", "rbf"), "C": [0.1, 1, 10]}, cv=5)
cv = ShuffleSplit(nbSplit, test_size=0.2, random_state=21)


# %% parameters to be applied to extract the features
#events = ["HC", "EP1"]#, "EP2", "EP3", "EP4"]

#events_id={"HC": 0, "EP1": 1, "EP2":2, "EP3":3, "EP4": 4}
freqbands = {'theta-alpha': [3, 14],  # Epilepsy case
             'paper': [3, 40]}

opt_trial_duration = [100864/256] # one single trial for everything
fs = 256

test=mat73.loadmat('/Users/marieconstance.corsi/Documents/GitHub/Fenicotteri-equilibristi/Database/1_Clinical/Epilepsy_GMD/Data_Epi_MEG_4Classif_concat_NoTrials.mat')
ch_names = test['labels_AAL1']
ch_types = ["eeg" for i in range(np.shape(ch_names)[0])]

#%%  load data & concatenate infos
precomp_name = path_data_root + '/PLV_HC_EPI1_concat_4Classif.mat'
test_plv = mat73.loadmat(precomp_name) # grp, pz, freqband

grp_id = ["HC", "EP1"]
data_theta = dict()
data_broad = dict()
database_concat_theta = dict()

idx = 0 # HC
for kk_pz in range(32):
    data_theta[kk_pz] = test_plv["PLV"][idx][kk_pz][0]
    data_broad[kk_pz] = test_plv["PLV"][idx][kk_pz][1]
for kk_pz in range(32): # EP1
    data_theta[31+kk_pz] = test_plv["PLV"][idx+1][kk_pz][0]
    data_broad[31+kk_pz] = test_plv["PLV"][idx+1][kk_pz][1]


data = dict()
data["theta-alpha"]= data_theta
data["paper"] = data_broad

labels = dict()
labels["theta-alpha"] = np.concatenate(([0]*32,[1]*32))
labels["paper"] = np.concatenate(([0]*32,[1]*32))

database_concat = dict()
database_concat["labels"] = labels
database_concat["data"] = data

#%% test 1 - classification HC vs EP1 - 32 vs 32
grp_id_2use = ['HC', 'EPI 1']

results_plv = pd.DataFrame()

for f in freqbands:
    fmin = freqbands[f][0]
    fmax = freqbands[f][1]
    temp_data_plv = database_concat["data"][f]
    temp_labels = database_concat["labels"][f]

    orderedNames = temp_data_plv.keys()
    data_plv = np.array([temp_data_plv[i] for i in orderedNames]) # transforms a dict to matrix

    for iperm in range(n_perms): # in case we want to do some permutations to increase the statistical power
        perm = rng.permutation(len(temp_data_plv))
        data_shuffle = data_plv[perm]
        #perm2 = rng.permutation(len(temp_data_plv)) # test
        label_shuffle = [temp_labels[x] for x in perm]
        nb_trials = len(data_shuffle)  # nb subjects/ pz here
        nb_ROIs = np.shape(data_shuffle)[1]


        PLV_cl = np.reshape(data_shuffle, (nb_trials, nb_ROIs * nb_ROIs))
        PLV_nodal_cl = np.sum(data_shuffle,1)

        clf_2 = Pipeline([('SVM', svm)])

        scores_PLV_SVM = cross_validate(clf_2, PLV_cl,  label_shuffle, cv=cv, n_jobs=None, scoring=scoring, return_estimator=False) # returns the estimator objects for each cv split!
        scores_PLV_nodal_SVM = cross_validate(clf_2, PLV_nodal_cl,  label_shuffle, cv=cv, n_jobs=None, scoring=scoring, return_estimator=False) # returns the estimator objects for each cv split!

        temp_results_plv = _build_df_plv_from_scores(scores_PLV_SVM, "PLV-matlab", nbSplit, fmin, fmax)
        temp_results_plv_nodal = _build_df_plv_from_scores(scores_PLV_nodal_SVM, "PLV-matlab-nodal", nbSplit, fmin, fmax)

        results_plv = pd.concat((results_plv, temp_results_plv, temp_results_plv_nodal))

    # concatenate results in a single dataframe
    results_global = results_plv
    results_global.to_csv(
            path_csv_root + "/SVM/HC_EP1_IndivOpt_PLV_matlab_SVM_Classification-allnode-2class-" +
            "-freq-" + str(fmin) + '-' + str(fmax) + '-nbSplit' + str(nbSplit) + ".csv"
        )
    print(
            "saved " +
            path_csv_root + "/SVM/HC_EP1_IndivOpt_PLV_matlab_SVM_Classification-allnode-2class-" +
            "-freq-" + str(fmin) + '-' + str(fmax) + '-nbSplit' + str(nbSplit) + ".csv"
        )

