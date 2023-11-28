# %%
"""
==============================================================
Attempt to classify MEG data in the source space - neuronal avalanches vs classical approaches - classification on longer trials w/ SVM
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
from mne import create_info, EpochsArray
from mne.decoding import CSP as CSP_MNE

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import ShuffleSplit, cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score

import numpy as np
from scipy.stats import zscore
from moabb.paradigms import MotorImagery

# to compute ImCoh estimations for each trial
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

def transprob(aval, nregions):  # (t,r)
    mat = np.zeros((nregions, nregions))
    norm = np.sum(aval, axis=0)
    for t in range(len(aval) - 1):
        ini = np.where(aval[t] == 1)
        mat[ini] += aval[t + 1]
    mat[norm != 0] = mat[norm != 0] / norm[norm != 0][:, None]
    return mat


def Transprob(ZBIN, nregions, val_duration):
    mat = np.zeros((nregions, nregions))
    A = np.sum(ZBIN, axis=1)
    a = np.arange(len(ZBIN))
    idx = np.where(A != 0)[0]
    aout = np.split(a[idx], np.where(np.diff(idx) != 1)[0] + 1)
    ifi = 0
    for iaut in range(len(aout)):
        if len(aout[iaut]) > val_duration:
            mat += transprob(ZBIN[aout[iaut]], nregions)
            ifi += 1
    mat = mat / ifi
    return mat, aout


def threshold_mat(data, thresh=3):
    current_data = data
    binarized_data = np.where(np.abs(current_data) > thresh, 1, 0)
    return (binarized_data)


def find_avalanches(data, thresh=3, val_duration=2):
    binarized_data = threshold_mat(data, thresh=thresh)
    N = binarized_data.shape[0]
    mat, aout = Transprob(binarized_data.T.values, N, val_duration)
    aout = np.array(aout, dtype=object)
    list_length = [len(i) for i in aout]
    unique_sizes = set(list_length)
    min_size, max_size = min(list_length), max(list_length)
    list_avalanches_bysize = {i: [] for i in unique_sizes}
    for s in aout:
        n = len(s)
        list_avalanches_bysize[n].append(s)
    return (aout, min_size, max_size, list_avalanches_bysize, mat)

def _compute_normalized_weights(scores, nbSplit, path2store):
    # retrieve estimators for each cv and compute median weight over the splits - compute absolute weights & normalize them
    weights_estim = pd.DataFrame()
    temp = dict()
    for kk_spl in range(nbSplit):
        temp2norm = list(np.abs(scores["estimator"][kk_spl].named_steps['SVM'].coef_[0]))
        temp_normalized = (temp2norm - np.min(temp2norm)) / (
                    np.max(temp2norm) - np.min(temp2norm))
        temp["split-" + str(kk_spl)] = temp_normalized / sum(temp_normalized)

    weights_estim = pd.DataFrame.from_dict(temp)
    weights_estim["median"] = np.median(weights_estim.values, 1)

    # save all the normalized weights & the associated median
    weights_estim.to_csv(path2store)
    print(
        "saved " + path2store
    )
    return weights_estim

def _build_df_atm_from_scores(scores, ppl_name, nbSplit, zthresh_value, aval_duration_value, fmin, fmax):
    pd_ATM_classif_res = pd.DataFrame.from_dict(scores)
    ppl_atm = [ppl_name] * len(pd_ATM_classif_res)
    pd_ATM_classif_res["pipeline"] = ppl_atm
    pd_ATM_classif_res["split"] = [nbSplit] * len(ppl_atm)
    pd_ATM_classif_res["zthresh"] = [zthresh_value] * len(ppl_atm)
    pd_ATM_classif_res["val_duration"] = [aval_duration_value] * len(ppl_atm)
    pd_ATM_classif_res["freq"] = [str(fmin) + '-' + str(fmax)] * len(ppl_atm)

    return  pd_ATM_classif_res

def _build_df_ImCoh_from_scores(scores, ppl_name, nbSplit, fmin, fmax):
    pd_ImCoh_classif_res = pd.DataFrame.from_dict(scores)
    ppl_ImCoh = [ppl_name] * len(pd_ImCoh_classif_res)
    pd_ImCoh_classif_res["pipeline"] = ppl_ImCoh
    pd_ImCoh_classif_res["split"] = [nbSplit] * len(ppl_ImCoh)
    pd_ImCoh_classif_res["freq"] = [str(fmin) + '-' + str(fmax)] * len(ppl_ImCoh)

    return  pd_ImCoh_classif_res

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
#svm = GridSearchCV(SVC(), {"kernel": ("linear", "rbf"), "C": [0.1, 1, 10]}, cv=5)
cv = ShuffleSplit(nbSplit, test_size=0.2, random_state=21)


# %% parameters to be applied to extract the features
freqbands = {'theta': [3, 8],
             'alpha': [8, 14],
             'theta-alpha': [3, 14],  # Epilepsy case
             'paper': [3, 40]}

opt_trial_duration = [100864/256] # one single trial for everything
fs = 256

test=mat73.loadmat('/Users/marieconstance.corsi/Documents/GitHub/Fenicotteri-equilibristi/Database/1_Clinical/Epilepsy_GMD/Data_Epi_MEG_4Classif_concat_NoTrials.mat')
ch_names = test['labels_AAL1']
ch_types = ["eeg" for i in range(np.shape(ch_names)[0])]

#%% Classification HC vs EP1 - 31 vs 31
grp_id_2use = ['HC', 'EPI 1']
precomp_concat_name = path_data_root + '/concat_epochs_HC_EP1.gz'

if osp.exists(precomp_concat_name):
    print("Loading existing concatenated precomputations...")
    with gzip.open(precomp_concat_name, "rb") as file:
        epochs_concat, labels_concat = pickle.load(file)
else:
    print("Please consider performing precomputations and/or concatenation of the epochs!")

epochs_concat['theta'] = epochs_concat['theta-alpha']
epochs_concat['alpha'] = epochs_concat['paper']
labels_concat['theta'] = labels_concat['theta-alpha']
labels_concat['alpha'] = labels_concat['paper']


for f in freqbands:
    fmin = freqbands[f][0]
    fmax = freqbands[f][1]
    epochs2use=epochs_concat[f].drop([2, 31, 32, 33, 34], reason='USER') # to have the same nb of pz/subjects and because data from the second patient may be corrupted
    results_ImCoh = pd.DataFrame()
    for iperm in range(n_perms): # in case we want to do some permutations to increase the statistical power
        perm = rng.permutation(len(epochs2use))
        epochs_in_permuted_order = epochs2use[perm]
        data_shuffle = epochs_in_permuted_order.get_data()
        label_shuffle = [labels_concat[f][x] for x in perm]
        nb_trials = len(data_shuffle)  # nb subjects/ pz here
        nb_ROIs = np.shape(data_shuffle)[1]

        ### ImCoh + SVM
        # here to compute FC on long recordings we increase the time window of interest to speed up the process
        ft = FunctionalTransformer(
            delta=30, ratio=0.5, method="imcoh", fmin=fmin, fmax=fmax
        )
        preproc_meg = Pipeline(steps=[("ft", ft)])  # , ("spd", EnsureSPD())])
        mat_ImCoh = preproc_meg.fit_transform(epochs_in_permuted_order)  # actually unfair because does an average of the estimations ie more robust than ATM this way

        ImCoh_cl = np.reshape(mat_ImCoh, (nb_trials, nb_ROIs * nb_ROIs))
        ImCoh_nodal_cl = np.sum(mat_ImCoh,1)

        # here just to find out the best hyperparameters, nothing more - need to be linear to retrieve features
        clf_prelim = GridSearchCV(SVC(kernel="linear"), {"C": [0.1, 1, 10]}, cv=5)
        clf_prelim.fit(ImCoh_cl, label_shuffle)
        best_C_ImCoh = clf_prelim.best_params_["C"]

        clf_prelim.fit(ImCoh_nodal_cl, label_shuffle)
        best_C_ImCoh_nodal = clf_prelim.best_params_["C"]

        # real classification now
        my_svm_ImCoh = SVC(C=best_C_ImCoh, kernel="linear")
        my_svm_ImCoh_nodal = SVC(C=best_C_ImCoh_nodal, kernel="linear")

        clf_ImCoh = Pipeline([('SVM', my_svm_ImCoh)])
        clf_ImCoh_nodal = Pipeline([('SVM', my_svm_ImCoh_nodal)])

        # score_ImCoh_SVM = cross_val_score(clf_2, ImCoh_cl, labels, cv=cv, n_jobs=None)
        scores_ImCoh_SVM = cross_validate(clf_ImCoh, ImCoh_cl,  label_shuffle, cv=cv, n_jobs=None, scoring=scoring, return_estimator=True) # returns the estimator objects for each cv split!
        scores_ImCoh_nodal_SVM = cross_validate(clf_ImCoh_nodal, ImCoh_nodal_cl,  label_shuffle, cv=cv, n_jobs=None, scoring=scoring, return_estimator=True) # returns the estimator objects for each cv split!

        # compute and store normalized weights
        path2store = path_csv_root + "/SVM/ImCoh_nodal_weights_HC_EP1_SVM_Classification" + "-freq-" + str(
            fmin) + '-' + str(fmax) + '-nbSplit' + str(nbSplit) + ".csv"
        weights_estim_ImCoh_nodal = _compute_normalized_weights(scores_ImCoh_nodal_SVM, nbSplit, path2store)

        path2store = path_csv_root + "/SVM/ImCoh_edges_weights_HC_EP1_SVM_Classification" + "-freq-" + str(
            fmin) + '-' + str(fmax) + '-nbSplit' + str(nbSplit) + ".csv"
        weights_estim_ImCoh = _compute_normalized_weights(scores_ImCoh_SVM, nbSplit, path2store)

        # concatenate ImCoh results in a dedicated dataframe
        temp_results_ImCoh_nodal = _build_df_ImCoh_from_scores(scores=scores_ImCoh_nodal_SVM, ppl_name="ImCoh+SVM-nodal",
                                                           nbSplit=nbSplit, fmin=fmin, fmax=fmax)
        temp_results_ImCoh = _build_df_ImCoh_from_scores(scores=scores_ImCoh_SVM, ppl_name="ImCoh+SVM",
                                                     nbSplit=nbSplit, fmin=fmin, fmax=fmax)
        results_ImCoh = pd.concat((temp_results_ImCoh, temp_results_ImCoh_nodal), ignore_index=True)

    # concatenate results in a single dataframe 
    results_global = results_ImCoh
    results_global.to_csv(
            path_csv_root + "/SVM/FeaturesInfos_HC_EP1_IndivOpt_ImCoh_Comparison_SVM_Classification-allnode-2class-" +
            "-freq-" + str(fmin) + '-' + str(fmax) + '-nbSplit' + str(nbSplit) + ".csv"
        )
    print(
            "saved " +
            path_csv_root + "/SVM/FeaturesInfos_HC_EP1_IndivOpt_ImCoh_Comparison_SVM_Classification-allnode-2class-" +
            "-freq-" + str(fmin) + '-' + str(fmax) + '-nbSplit' + str(nbSplit) + ".csv"
        )

#%% List of labels - Desikan
path_figures_root = "/Users/marieconstance.corsi/Documents/GitHub/Fenicotteri-equilibristi/Figures/Classification/"

freqbands = {'theta': [3, 8],
             'alpha': [8, 14],
             'theta-alpha': [3, 14],  # Epilepsy case
             'paper': [3, 40]}

labels = pd.read_csv(path_csv_root + "LabelsDesikanAtlas.csv")
df_weights_estim_ImCoh_nodal = dict()
df_weights_estim_ImCoh_edges = dict()
df_weights_estim_ImCoh_edges_nodal = dict()
df_weights_estim_ImCoh_edges_nodal_max = dict()
df_weights_estim_ImCoh_edges_nodal_mean = dict()

index_values = labels.values
column_values = labels.values

for f in freqbands:
    fmin = freqbands[f][0]
    fmax = freqbands[f][1]

    temp = pd.read_csv(path_csv_root + "/SVM/ImCoh_nodal_weights_HC_EP1_SVM_Classification" + "-freq-" + str(
        fmin) + '-' + str(fmax) + '-nbSplit' + str(nbSplit) + ".csv")
    weights_estim_ImCoh_nodal = temp["median"]
    df_weights_estim_ImCoh_nodal[f] = pd.DataFrame(data = weights_estim_ImCoh_nodal)
    temp2 = pd.read_csv(path_csv_root + "/SVM/ImCoh_edges_weights_HC_EP1_SVM_Classification" + "-freq-" + str(
        fmin) + '-' + str(fmax) + '-nbSplit' + str(nbSplit) + ".csv")
    weights_estim_ImCoh_edges = np.array(temp2["median"]).reshape(68,68)
    df_weights_estim_ImCoh_edges[f] = pd.DataFrame(data = weights_estim_ImCoh_edges,
                                              index=index_values,
                                              columns=column_values)

    weights_estim_ImCoh_edges_nodal = np.median(weights_estim_ImCoh_edges, 1)
    df_weights_estim_ImCoh_edges_nodal[f] = pd.DataFrame(data = weights_estim_ImCoh_edges_nodal,
                                              index=index_values,
                                              columns=["ROIs"])

    weights_estim_ImCoh_edges_nodal_mean = np.mean(weights_estim_ImCoh_edges, 1)
    df_weights_estim_ImCoh_edges_nodal_mean[f] = pd.DataFrame(data = weights_estim_ImCoh_edges_nodal_mean,
                                              index=index_values,
                                              columns=["ROIs"])
    weights_estim_ImCoh_edges_nodal_max = np.mean(weights_estim_ImCoh_edges, 1)
    df_weights_estim_ImCoh_edges_nodal_max[f] = pd.DataFrame(data = weights_estim_ImCoh_edges_nodal_max,
                                              index=index_values,
                                              columns=["ROIs"])
#%% save .mat to plot nodal results into scalp
features = {"df_weights_estim_ImCoh_edges_nodal_theta":df_weights_estim_ImCoh_edges_nodal['theta']['ROIs'].values,
            "df_weights_estim_ImCoh_nodal_theta": df_weights_estim_ImCoh_nodal['theta'].values,
            "df_weights_estim_ImCoh_edges_nodal_alpha":df_weights_estim_ImCoh_edges_nodal['alpha']['ROIs'].values,
            "df_weights_estim_ImCoh_nodal_alpha": df_weights_estim_ImCoh_nodal['alpha'].values,
            "df_weights_estim_ImCoh_edges_nodal_theta_alpha":df_weights_estim_ImCoh_edges_nodal['theta-alpha']['ROIs'].values,
            "df_weights_estim_ImCoh_nodal_theta_alpha": df_weights_estim_ImCoh_nodal['theta-alpha'].values,
            "df_weights_estim_ImCoh_edges_nodal_theta_alpha_mean": df_weights_estim_ImCoh_edges_nodal_mean['theta-alpha'][
                            'ROIs'].values,
            "df_weights_estim_ImCoh_edges_nodal_broad": df_weights_estim_ImCoh_edges_nodal['paper'][
                'ROIs'].values,
            "df_weights_estim_ImCoh_edges_nodal_broad_mean": df_weights_estim_ImCoh_edges_nodal_mean['paper'][
                            'ROIs'].values,
            "df_weights_estim_ImCoh_edges_nodal_broad_max": df_weights_estim_ImCoh_edges_nodal_max['paper'][
                            'ROIs'].values,
            "df_weights_estim_ImCoh_nodal_broad": df_weights_estim_ImCoh_nodal['paper'].values,
            "df_weights_estim_ImCoh_edges_broad": df_weights_estim_ImCoh_edges['paper'].T.values,
            "df_weights_estim_ImCoh_edges_theta": df_weights_estim_ImCoh_edges['theta'].T.values,
            "df_weights_estim_ImCoh_edges_alpha": df_weights_estim_ImCoh_edges['alpha'].T.values,
            "df_weights_estim_ImCoh_edges_theta_alpha": df_weights_estim_ImCoh_edges['theta-alpha'].T.values,

            }
scipy.io.savemat(path_csv_root+'Features_ImCoh.mat',features)