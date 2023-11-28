# %%
"""
==============================================================
Attempt to classify MEG data in the source space - neuronal avalanches vs classical approaches - anomaly detection HC vs EP1

A. Novelty detection
    A.1. Novelty w/ One-class SVM - not a real evaluation performance, gives just the number of errors..

        # Novelty w/ One-class SVM with non-linear kernel (RBF)
        #https://scikit-learn.org/stable/auto_examples/svm/plot_oneclass.html#sphx-glr-auto-examples-svm-plot-oneclass-py

        # One-Class SVM versus One-Class SVM using Stochastic Gradient Descent
        #https://scikit-learn.org/stable/auto_examples/linear_model/plot_sgdocsvm_vs_ocsvm.html#sphx-glr-auto-examples-linear-model-plot-sgdocsvm-vs-ocsvm-py

    A.2. Novelty w/ Local Outlier Factor
        #https://scikit-learn.org/stable/modules/outlier_detection.html#novelty-detection-with-local-outlier-factor

B. Outlier detection - assumption: anomalies are rare (ie more for unbalanced cases)

===============================================================

"""
# Authors: Marie-Constance Corsi <marie.constance.corsi@gmail.com>
#
# License: BSD (3-clause)


import os.path as osp
import os

import pandas as pd

import mat73

import matplotlib
import matplotlib.pyplot as plt

import mne

import numpy as np

import pickle
import gzip

from scipy.stats import zscore

from sklearn.model_selection import ShuffleSplit
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import SGDOneClassSVM
from sklearn.pipeline import make_pipeline
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor


font = {"weight": "normal", "size": 15}

matplotlib.rc("font", **font)

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
    mat, aout = Transprob(binarized_data.T, N, val_duration)
    aout = np.array(aout, dtype=object)
    list_length = [len(i) for i in aout]
    unique_sizes = set(list_length)
    min_size, max_size = min(list_length), max(list_length)
    list_avalanches_bysize = {i: [] for i in unique_sizes}
    for s in aout:
        n = len(s)
        list_avalanches_bysize[n].append(s)
    return (aout, min_size, max_size, list_avalanches_bysize, mat)

#%% parameters for novelty detection
cv = ShuffleSplit(n_splits=10, test_size=.25, random_state=0)
random_state = 42
rng = np.random.RandomState(random_state)

# OCSVM hyperparameters
nu = 0.05
gamma = 2.0
clf = OneClassSVM(gamma=gamma, kernel="rbf", nu=nu)

# 1-SVM + kernel
transform = Nystroem(gamma=gamma, random_state=random_state, n_components=10) # default n_components=100
clf_sgd = SGDOneClassSVM(
    nu=nu, shuffle=True, fit_intercept=True, random_state=random_state, tol=1e-4
)

# LOF
amount_contamination = 'auto'  # 0.01 # cf novelty approach
pipe_lof = LocalOutlierFactor(n_neighbors=15, novelty=True, algorithm='auto', leaf_size=30,
                              metric='minkowski', p=2, contamination=amount_contamination) # 20 neighbors ok, to be increased up to 35 if high density of outliers - the rest : default parameters


#%% parameters to be applied to extract the features
#events = ["HC", "EP1"]#, "EP2", "EP3", "EP4"]

#events_id={"HC": 0, "EP1": 1, "EP2":2, "EP3":3, "EP4": 4}
freqbands = {'theta-alpha': [3, 14],  # Epilepsy case
             'paper': [3, 40]}
# opt_zthresh = [2]  # 1.4
# opt_val_duration = [2]
opt_zthresh = [1.4, 1.6, 1.8, 2, 2.2, 2.4, 2.6, 2.8, 3]
opt_val_duration = [2, 3, 4, 5, 6, 7, 8]

opt_trial_duration = [100864/256] # one single trial for everything
fs = 256

test=mat73.loadmat('/Users/marieconstance.corsi/Documents/GitHub/Fenicotteri-equilibristi/Database/1_Clinical/Epilepsy_GMD/Data_Epi_MEG_4Classif_concat_NoTrials.mat')
ch_names = test['labels_AAL1']
ch_types = ["eeg" for i in range(np.shape(ch_names)[0])]

# %% TODO: done once - beware, huge files (>5Go) - load data from matlab & transform data to mne raw data, filter  and cut them into epochs - to be done once to save space
# precomp_name = path_data_root + '/epochs_HC_EPx.gz'
#
# if osp.exists(precomp_name):
#     print("Loading existing precomputations...")
#     with gzip.open(precomp_name, "rb") as file:
#         epochs, labels = pickle.load(file)
# else:
#
#     data_full = mat73.loadmat(
#         '/Users/marieconstance.corsi/Documents/GitHub/Fenicotteri-equilibristi/Database/1_Clinical/Epilepsy_GMD/MEG_data_Infos_Epi_HC_Ex_concat_full.mat')
#     grp_id = data_full["gr_names"]
#     data = dict()
#     for idx, g_id in enumerate(grp_id):
#         temp = data_full["FULL_concat"][idx]
#         data_pz = dict()
#         for pz in range(len(temp)):
#             temp_pz = temp[pz]["serie_temporali"]
#             data_pz[pz] = temp_pz
#         data[g_id] = data_pz
#
#     labels = dict()
#     epochs = dict()
#     for f in freqbands:
#         fmin = freqbands[f][0]
#         fmax = freqbands[f][1]
#         for kk_trial_duration in opt_trial_duration: # for the moment we work with one single and long trial per pz/subject
#             temp_labels = dict()
#             temp_epochs = dict()
#             for idx, g_id in enumerate(grp_id):
#                 nb_pz = len(data[g_id])
#                 epochs_pz = dict()
#                 labels_pz = dict()
#                 for pz in range(nb_pz):
#                     x = data[g_id][pz]  # 2D: nb rois x nb_samples
#                     info = mne.create_info(ch_names, fs, ch_types)
#                     raw = mne.io.RawArray(data=np.array(x), info=info)
#
#                     # filter data with the freqband
#                     raw.filter(fmin, fmax, fir_design="firwin", skip_by_annotation="edge")
#                     # identify events
#                     new_events = mne.make_fixed_length_events(raw, start=0, stop=None, duration=kk_trial_duration, overlap=0)
#                     # put as event_id the number associated to the grp
#                     new_events[:, -1] = idx*new_events[:, -1]
#                     epochs_pz[pz]=mne.Epochs(raw, new_events, tmin=0, tmax=kk_trial_duration, baseline=None)
#                     labels_pz[pz] = epochs_pz[pz].events[:, -1]
#                 temp_epochs[g_id] = epochs_pz
#                 temp_labels[g_id] = labels_pz
#             epochs[f] = temp_epochs
#             labels[f] = temp_labels
#     with gzip.open(precomp_name, "w") as file:
#         pickle.dump([epochs, labels], file)
#         print('Precomputed epochs and labels saved')
#
#
# #%% concatenate epochs to do HC vs EP1
#
# grp_id_2use = ['HC', 'EPI 1']
# precomp_concat_name = path_data_root + '/concat_epochs_HC_EP1.gz'
#
# if osp.exists(precomp_concat_name):
#     print("Loading existing concatenated precomputations...")
#     with gzip.open(precomp_concat_name, "rb") as file:
#         epochs_concat, labels_concat = pickle.load(file)
# else:
#     print("Computing concatenations...")
#     epochs_concat = dict()
#     labels_concat = dict()
#     for f in freqbands:
#         fmin = freqbands[f][0]
#         fmax = freqbands[f][1]
#
#         results_global = pd.DataFrame()
#         ep = epochs[f]
#         lab = labels[f]
#         for kk_trial_duration in opt_trial_duration:
#             # concatenate epochs to work with afterwards - TODO: update idx with new shapes
#             temp_epochs_concat = ep[grp_id[0]][0] # initialization
#             temp_labels_concat = [0] # initialization
#             for idx, g_id in enumerate(grp_id_2use):
#                 epochs_pz = ep[g_id]
#                 nb_pz = len(epochs_pz)  # 32 # to have the same number of pz # len(data[g_id])
#                 for pz in range(1, nb_pz):
#                     temp_epochs_concat = mne.concatenate_epochs([temp_epochs_concat,epochs_pz[pz]], verbose=False)
#                     temp_labels_concat = np.concatenate ((temp_labels_concat, epochs_pz[pz].events[:, -1]))
#             epochs_concat[f] = temp_epochs_concat
#             labels_concat[f] = temp_labels_concat
#     with gzip.open(precomp_concat_name, "w") as file:
#         pickle.dump([epochs_concat, labels_concat], file)
#         print('Precomputed & concatenated epochs and labels saved')

#%% test 1 - Novelty from HC (used for training) vs EP1 - 32 vs 32
grp_id_2use = ['HC', 'EPI 1']
precomp_concat_name = path_data_root + '/concat_epochs_HC_EP1.gz'
path_figures_root = "/Users/marieconstance.corsi/Documents/GitHub/Fenicotteri-equilibristi/Figures/Classification/"

if osp.exists(precomp_concat_name):
    print("Loading existing concatenated precomputations...")
    with gzip.open(precomp_concat_name, "rb") as file:
        epochs_concat, labels_concat = pickle.load(file)
else:
    print("Please consider performing precomputations and/or concatenation of the epochs!")


HC_EP1_novelty_res = list()
### ATM + SVM -- attempt to optimize the code to make it faster...
for f in freqbands:
    fmin = freqbands[f][0]
    fmax = freqbands[f][1]
    epochs2use=epochs_concat[f].drop([32, 33, 34], reason='USER') # to have the same nb of pz/subjects
    nb_pz = 32

    data_all = epochs2use.get_data()
    temp = np.transpose(data_all, (1, 0, 2))
    temp_nc = np.reshape(temp, (np.shape(temp)[0], np.shape(temp)[1] * np.shape(temp)[2]))
    zscored_data = zscore(temp_nc, axis=1)
    # epoching here before computing the avalanches
    temp_zscored_data_ep = np.reshape(zscored_data,
                                      (np.shape(temp)[0], np.shape(temp)[1], np.shape(temp)[2]))
    zscored_data_ep = np.transpose(temp_zscored_data_ep, (1, 0, 2))

    nb_trials = np.shape(data_all)[0]  # nb subjects/ pz here
    nb_ROIs = np.shape(data_all)[1]

    for kk_zthresh in opt_zthresh:
        for kk_val_duration in opt_val_duration:
            ATM = np.empty((nb_trials, nb_ROIs, nb_ROIs))
            for kk_trial in range(nb_trials):
                list_avalanches, min_size_avalanches, max_size_avalanches, list_avalanches_bysize, temp_ATM = find_avalanches(
                    zscored_data_ep[kk_trial, :, :], thresh=kk_zthresh, val_duration=kk_val_duration)
                # ATM: nb_trials x nb_ROIs x nb_ROIs matrix
                ATM[kk_trial, :, :] = temp_ATM
            reshape_ATM = np.reshape(ATM, (np.shape(ATM)[0], np.shape(ATM)[1] * np.shape(ATM)[2]))
            fixed_reshape_ATM = np.nan_to_num(reshape_ATM, nan=0)

            y_hc_cv = labels_concat[f][:nb_pz]
            X_hc_ = fixed_reshape_ATM[:nb_pz, :]
            y_hc_ = y_hc_cv

            for kk_pz in range(nb_pz):  # one by one, to be optimzed? # TODO: si fonctionne faire un autre split pour les novels à considérer
                X_outliers = [fixed_reshape_ATM[nb_pz+kk_pz]]
                res_info = {
                    "pz_novel": kk_pz,
                    "n_sessions": 1,
                    "FreqBand": str(fmin) + '-' + str(fmax),
                    "dataset": "HC_EP1",
                    "zthresh": kk_zthresh,
                    "val_duration": kk_val_duration,

                }

                for idx, (train, test) in enumerate(cv.split(X_hc_, y_hc_)):
                    X_train = X_hc_[train]
                    X_test = X_hc_[test]

                    # Fit the One-Class SVM
                    clf.fit(X_train)
                    y_pred_train = clf.predict(X_train)
                    y_pred_test = clf.predict(X_test)
                    y_pred_outliers = clf.predict(X_outliers)
                    n_error_train = y_pred_train[y_pred_train == -1].size
                    n_error_test = y_pred_test[y_pred_test == -1].size
                    n_error_outliers = y_pred_outliers[y_pred_outliers == 1].size

                    # estim number of errors each time
                    res = {
                        "pipeline": 'OneClassSVM_RBF',
                        "n_error_train": n_error_train,
                        "n_error_test": n_error_test,
                        "n_error_outliers": n_error_outliers,
                        "samples": len(y_pred_test),
                        "time": 0.0,
                        "split": idx,
                        **res_info,
                    }
                    HC_EP1_novelty_res.append(res)

                    # Fit the One-Class SVM using a kernel approximation and SGD
                    pipe_sgd = make_pipeline(transform, clf_sgd)
                    pipe_sgd.fit(X_train)
                    y_pred_train_sgd = pipe_sgd.predict(X_train)
                    y_pred_test_sgd = pipe_sgd.predict(X_test)
                    y_pred_outliers_sgd = pipe_sgd.predict(X_outliers)
                    n_error_train_sgd = y_pred_train_sgd[y_pred_train_sgd == -1].size
                    n_error_test_sgd = y_pred_test_sgd[y_pred_test_sgd == -1].size
                    n_error_outliers_sgd = y_pred_outliers_sgd[y_pred_outliers_sgd == 1].size

                    res = {
                        "pipeline": 'OneClassSVM_kernel',
                        "n_error_train": n_error_train_sgd,
                        "n_error_test": n_error_test_sgd,
                        "n_error_outliers": n_error_outliers_sgd,
                        "samples": len(y_pred_test_sgd),
                        "time": 0.0,
                        "split": idx,
                        **res_info,
                    }
                    HC_EP1_novelty_res.append(res)

                    # Use of Local Outlier Factor
                    pipe_lof.fit(X_train)
                    y_pred_train_lof = pipe_lof.predict(X_train)
                    y_pred_test_lof = pipe_lof.predict(X_test)
                    y_pred_outliers_lof = pipe_lof.predict(X_outliers)
                    n_error_train_lof = y_pred_train_lof[y_pred_train_lof == -1].size
                    n_error_test_lof = y_pred_test_lof[y_pred_test_lof == -1].size
                    n_error_outliers_lof = y_pred_outliers_lof[y_pred_outliers_lof == 1].size

                    res = {
                        "pipeline": 'LocalOutlierFactor',
                        "n_error_train": n_error_train_lof,
                        "n_error_test": n_error_test_lof,
                        "n_error_outliers": n_error_outliers_lof,
                        "samples": len(y_pred_test_lof),
                        "time": 0.0,
                        "split": idx,
                        **res_info,
                    }
                    HC_EP1_novelty_res.append(res)

                # we plot only the last results - ie last split
                Z = clf.decision_function(X_test)
                Z_sgd = pipe_sgd.decision_function(X_test)
                Z_lof = pipe_lof.decision_function(X_test)


HC_EP1_novelty_res = pd.DataFrame(HC_EP1_novelty_res)
HC_EP1_novelty_res.to_csv(path_csv_root +
            "ATM_Novelty_LOC_IndivPlot-allnode-2class-HC_vs_EP1_pz.csv"
)
print(
    "saved "
    + path_csv_root
    + "ATM_Novelty_LOC_IndivPlot-allnode-2class-HC_vs_EP1_pz.csv"
)
#%% TODO : for the best configuration - plots 2D ?

#%% Outlier detection - assumption that anomalies are rare ie only for unbalanced cases (HC vs EP4...)
# TODO: faire script à part au propre & read:
#  https://scikit-learn.org/stable/modules/outlier_detection.html
#  https://machinelearningmastery.com/model-based-outlier-detection-and-removal-in-python/
#  (en priorité) https://www.slideshare.net/agramfort/anomalynovelty-detection-with-scikitlearn?from_action=save

# TODO: plot ROC AUC
#%% Novelty detection
# TODO: faire script à part au propre & read
#%%
    # TODO: between groups - 1 big trial:
        # TODO: SVM :
            #  - sensible au nombre de trials ie si décide que sur 1 enregistrement marchera moins que novelty & outlier approaches
            # - quelle baseline/benchmark???


        # TODO: novelty (CSP & ATM)
            # - training sur les sujets sains et ajout d'1 patient  => loop à faire
        # TODO: outlier (CSP & ATM)
            # - training avec autant de sujets sains que de patients => idem pr testing

    # TODO: within groups voire within patient
        # TODO: outlier - EPI ie si tronçons de trials avec signature particulière et si oui laquelle (durée aval, taille aval, régions...)


    # TODO: voir contenu random forst & explications


    # TODO: identifier durée min d'enregistrement pour pouvoir détecter activité "anormale" - visée clinique

#%%
