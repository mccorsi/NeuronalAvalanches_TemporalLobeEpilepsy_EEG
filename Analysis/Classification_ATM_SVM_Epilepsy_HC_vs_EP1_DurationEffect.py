# %%
"""
==============================================================================================
Study of the effect of the duration of the signal on the classification performance
===============================================================================================

"""
# Authors: Marie-Constance Corsi <marie.constance.corsi@gmail.com>
#
# License: BSD (3-clause)

import gzip
import mat73

import numpy as np

import os.path as osp
import os

import pandas as pd
import pickle

import random

from scipy.stats import zscore

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_validate


# %%
if os.path.basename(os.getcwd()) == "NeuronalAvalanches_TemporalLobeEpilepsy_EEG":
    os.chdir("Database/1_Clinical/Epilepsy_GMD/")
if os.path.basename(os.getcwd()) == "Analysis":
    os.chdir("/Users/marieconstance.corsi/Documents/GitHub/NeuronalAvalanches_TemporalLobeEpilepsy_EEG/Database/1_Clinical/Epilepsy_GMD")
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


def _build_df_atm_from_scores(scores, ppl_name, nbSplit, zthresh_value, aval_duration_value, fmin, fmax, duration, permutation):
    pd_ATM_classif_res = pd.DataFrame.from_dict(scores)
    ppl_atm = [ppl_name] * len(pd_ATM_classif_res)
    pd_ATM_classif_res["pipeline"] = ppl_atm
    pd_ATM_classif_res["split"] = [nbSplit] * len(ppl_atm)
    pd_ATM_classif_res["zthresh"] = [zthresh_value] * len(ppl_atm)
    pd_ATM_classif_res["val_duration"] = [aval_duration_value] * len(ppl_atm)
    pd_ATM_classif_res["freq"] = [str(fmin) + '-' + str(fmax)] * len(ppl_atm)
    pd_ATM_classif_res["trial-duration"] = [duration] * len(ppl_atm)
    pd_ATM_classif_res["trial-permutation"] = [permutation] * len(ppl_atm)
    pd_ATM_classif_res["split"] = range(0,nbSplit)

    return  pd_ATM_classif_res

#%% parameters for classification
# kk_components = 8
nbSplit = 50
scoring = ['precision', 'recall', 'accuracy', 'f1', 'roc_auc']

# shuffle order of epochs & labels
rng = np.random.default_rng(42)  # set a random seed
n_perms = 1 # number of permutations wanted

# parameters for the default classifier & cross-validation
svm = GridSearchCV(SVC(), {"kernel": ("linear", "rbf"), "C": [0.1, 1, 10]}, cv=5)
cv = ShuffleSplit(nbSplit, test_size=0.2, random_state=21)


# %% parameters to be applied to extract the features
freqbands = {'theta-alpha': [3, 14],
             'paper': [3, 40]}

opt_trial_duration = [5, 10, 15, 30, 60, 120, 180, 300] # in seconds
fs = 256

test=mat73.loadmat('/Users/marieconstance.corsi/Documents/GitHub/NeuronalAvalanches_TemporalLobeEpilepsy_EEG/Database/1_Clinical/Epilepsy_GMD/Data_Epi_MEG_4Classif_concat_NoTrials.mat')
ch_names = test['labels_AAL1']
ch_types = ["eeg" for i in range(np.shape(ch_names)[0])]

#%% retrieve optimal parameters, for each freq band [zthresh, aval_dur]
opt_atm_param_edge = dict()
opt_atm_param_nodal = dict()
perf_opt_atm_param_edge = dict()
perf_opt_atm_param_nodal = dict()

df_res_opt_db=pd.read_csv(
    path_csv_root + "/SVM/OptConfig_HC_EP1_MEG_ATM_SVM_Classification-allnode_rest_BroadBand.csv"
)
opt_atm_param_edge["paper"] = [df_res_opt_db["zthresh"][0], df_res_opt_db["val_duration"][0]]
perf_opt_atm_param_edge["paper"] = df_res_opt_db["test_accuracy"][0]

df_res_opt_theta_alpha=pd.read_csv(
    path_csv_root + "/SVM/OptConfig_HC_EP1_MEG_ATM_SVM_Classification-allnode_rest_theta_alphaBand.csv"
)
opt_atm_param_edge["theta-alpha"] = [df_res_opt_theta_alpha["zthresh"][0], df_res_opt_theta_alpha["val_duration"][0]]
perf_opt_atm_param_edge["theta-alpha"] = df_res_opt_theta_alpha["test_accuracy"][0]

df_res_opt_theta=pd.read_csv(
    path_csv_root + "/SVM/OptConfig_HC_EP1_MEG_ATM_SVM_Classification-allnode_rest_theta_Band.csv"
)
opt_atm_param_edge["theta"] = [df_res_opt_theta["zthresh"][0], df_res_opt_theta["val_duration"][0]]
perf_opt_atm_param_edge["theta"] = df_res_opt_theta["test_accuracy"][0]

df_res_opt_alpha=pd.read_csv(
    path_csv_root + "/SVM/OptConfig_HC_EP1_MEG_ATM_SVM_Classification-allnode_rest_alphaBand.csv"
)
opt_atm_param_edge["alpha"] = [df_res_opt_alpha["zthresh"][0], df_res_opt_alpha["val_duration"][0]]
perf_opt_atm_param_edge["alpha"] = df_res_opt_alpha["test_accuracy"][0]


df_res_opt_db_nodal=pd.read_csv(
    path_csv_root + "/SVM/OptConfig_HC_EP1_MEG_ATM_SVM_Classification-nodal_rest_BroadBand.csv"
)
opt_atm_param_nodal["paper"] = [df_res_opt_db_nodal["zthresh"][0], df_res_opt_db_nodal["val_duration"][0]]
perf_opt_atm_param_nodal["paper"] = df_res_opt_db_nodal["test_accuracy"][0]

df_res_opt_theta_alpha_nodal=pd.read_csv(
    path_csv_root + "/SVM/OptConfig_HC_EP1_MEG_ATM_SVM_Classification-nodal_rest_theta_alphaBand.csv"
)
opt_atm_param_nodal["theta-alpha"] = [df_res_opt_theta_alpha_nodal["zthresh"][0], df_res_opt_theta_alpha_nodal["val_duration"][0]]
perf_opt_atm_param_nodal["theta-alpha"] = df_res_opt_theta_alpha_nodal["test_accuracy"][0]

df_res_opt_theta_nodal=pd.read_csv(
    path_csv_root + "/SVM/OptConfig_HC_EP1_MEG_ATM_SVM_Classification-nodal_rest_theta_Band.csv"
)
opt_atm_param_nodal["theta"] = [df_res_opt_theta_nodal["zthresh"][0], df_res_opt_theta_nodal["val_duration"][0]]
perf_opt_atm_param_nodal["theta"] = df_res_opt_theta_nodal["test_accuracy"][0]

df_res_opt_alpha_nodal=pd.read_csv(
    path_csv_root + "/SVM/OptConfig_HC_EP1_MEG_ATM_SVM_Classification-nodal_rest_alphaBand.csv"
)
opt_atm_param_nodal["alpha"] = [df_res_opt_alpha_nodal["zthresh"][0], df_res_opt_alpha_nodal["val_duration"][0]]
perf_opt_atm_param_nodal["alpha"] = df_res_opt_alpha_nodal["test_accuracy"][0]


#%% Classification HC vs EP1 - 31 vs 31
grp_id_2use = ['HC', 'EPI 1']
precomp_concat_name = path_data_root + '/concat_epochs_HC_EP1.gz'

if osp.exists(precomp_concat_name):
    print("Loading existing concatenated precomputations...")
    with gzip.open(precomp_concat_name, "rb") as file:
        epochs_concat, labels_concat = pickle.load(file)
else:
    print("Please consider performing precomputations and/or concatenation of the epochs!")

results_atm = pd.DataFrame()
num_perm = 100
max_time = 100864/256 # recording length
for f in freqbands:
    fmin = freqbands[f][0]
    fmax = freqbands[f][1]
    epochs2use = epochs_concat[f].drop([2, 31, 32, 33, 34],
                                       reason='USER')  # to have the same nb of pz/subjects and because data from the second patient may be corrupted
    for kk_trial_crop in opt_trial_duration:
        for kk_perm in range(num_perm):
            start = random.randrange(0, max_time - kk_trial_crop) # pick one random starting time between 0 and end - trial length to avoid issues
            tmin = start
            tmax = start + kk_trial_crop
            epochs_crop = epochs2use.copy().crop(tmin=tmin, tmax=tmax, include_tmax=True) # copy to avoid crop of the original epochs

            for iperm in range(n_perms): # in case we want to do some permutations to increase the statistical power
                perm = rng.permutation(len(epochs_crop))
                epochs_in_permuted_order = epochs_crop[perm]
                data_shuffle = epochs_in_permuted_order.get_data()
                label_shuffle = [labels_concat[f][x] for x in perm]
                nb_trials = len(data_shuffle)  # nb subjects/ pz here
                nb_ROIs = np.shape(data_shuffle)[1]

                ### ATM + SVM
                temp = np.transpose(data_shuffle, (1, 0, 2))
                temp_nc = np.reshape(temp, (np.shape(temp)[0], np.shape(temp)[1] * np.shape(temp)[2]))
                zscored_data = zscore(temp_nc, axis=1)
                # epoching here before computing the avalanches
                temp_zscored_data_ep = np.reshape(zscored_data,
                                                  (np.shape(temp)[0], np.shape(temp)[1], np.shape(temp)[2]))
                zscored_data_ep = np.transpose(temp_zscored_data_ep, (1, 0, 2))

                # get optimal parameters to make the process faster
                kk_zthresh_edge = opt_atm_param_edge[f][0]
                kk_val_duration_edge = opt_atm_param_edge[f][1]
                kk_zthresh_nodal = opt_atm_param_nodal[f][0]
                kk_val_duration_nodal = opt_atm_param_nodal[f][1]

                # ATM computation with edge-wise optimization parameters
                ATM = np.empty((nb_trials, nb_ROIs, nb_ROIs))
                for kk_trial in range(nb_trials):
                    list_avalanches, min_size_avalanches, max_size_avalanches, list_avalanches_bysize, temp_ATM = find_avalanches(
                        zscored_data_ep[kk_trial, :, :], thresh=kk_zthresh_edge, val_duration=kk_val_duration_edge)
                    # ATM: nb_trials x nb_ROIs x nb_ROIs matrix
                    ATM[kk_trial, :, :] = temp_ATM
                reshape_ATM = np.reshape(ATM, (np.shape(ATM)[0], np.shape(ATM)[1] * np.shape(ATM)[2]))
                fixed_reshape_ATM_cl = np.nan_to_num(reshape_ATM, nan=0)

                # ATM computation with node-wise optimization parameters
                ATM_nodal = np.empty((nb_trials, nb_ROIs, nb_ROIs))
                for kk_trial in range(nb_trials):
                    list_avalanches, min_size_avalanches, max_size_avalanches, list_avalanches_bysize, temp_ATM_nodal = find_avalanches(
                        zscored_data_ep[kk_trial, :, :], thresh=kk_zthresh_nodal, val_duration=kk_val_duration_nodal)
                    # ATM: nb_trials x nb_ROIs x nb_ROIs matrix
                    ATM_nodal[kk_trial, :, :] = temp_ATM_nodal
                temp_ATM_nod = np.nan_to_num(ATM_nodal, nan=0)
                ATM_nodal_cl = np.sum(temp_ATM_nod, 1)

                clf_2 = Pipeline([('SVM', svm)])
                scores_ATM_SVM = cross_validate(clf_2, fixed_reshape_ATM_cl, label_shuffle, cv=cv, n_jobs=None,
                                                scoring=scoring, return_estimator=False)
                scores_ATM_SVM_nodal = cross_validate(clf_2, ATM_nodal_cl, label_shuffle, cv=cv, n_jobs=None,
                                                      scoring=scoring, return_estimator=False)

                # concatenate ATM results in a dedicated dataframe
                temp_results_atm_nodal = _build_df_atm_from_scores(scores=scores_ATM_SVM_nodal,
                                                                   ppl_name="ATM+SVM-nodal",
                                                                   nbSplit=nbSplit, zthresh_value=kk_zthresh_nodal,
                                                                   aval_duration_value=kk_val_duration_nodal, fmin=fmin,
                                                                   fmax=fmax, duration=kk_trial_crop, permutation=kk_perm)
                temp_results_atm = _build_df_atm_from_scores(scores=scores_ATM_SVM, ppl_name="ATM+SVM",
                                                             nbSplit=nbSplit, zthresh_value=kk_zthresh_edge,
                                                             aval_duration_value=kk_val_duration_edge, fmin=fmin,
                                                             fmax=fmax, duration=kk_trial_crop, permutation=kk_perm)
                results_atm = pd.concat((results_atm, temp_results_atm, temp_results_atm_nodal))

# concatenate results in a single dataframe
results_global = results_atm
results_global.to_csv(
        path_csv_root + "/SVM/TrialDurationEffect_HC_EP1_IndivOpt_ATM__SVM_Classification-allnode-2class-" +
                '-nbSplit' + str(nbSplit) + ".csv"
        )
print(
        "saved " +
        path_csv_root + "/SVM/TrialDurationEffect_HC_EP1_IndivOpt_ATM__SVM_Classification-allnode-2class-" +
        '-nbSplit' + str(nbSplit) + ".csv"
        )

