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

import ptitprince as pt

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

# to compute ImCoh estimations for each trial
from Scripts.py_viz.fc_pipeline import FunctionalTransformer

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


# %% parameters for classification
# kk_components = 8
nbSplit = 50
scoring = ['precision', 'recall', 'accuracy', 'f1', 'roc_auc']
# recall = tp / (tp + fn) # default avg binary, to be changed if unbalanced/more classes - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html#sklearn.metrics.recall_score
# precision = tp / (tp + fp) # default avg binary, to be changed if unbalanced/more classes - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html#sklearn.metrics.precision_score
# f1 = F1 = 2 * (precision * recall) / (precision + recall) # here binary & balanced, otherwise need to be changed - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score
# roc_auc - Area under the receiver operating characteristic curve from prediction scores, default macro, to be changed for unbalanced and more classes - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html#sklearn.metrics.roc_auc_score

# shuffle order of epochs & labels
rng = np.random.default_rng(42)  # set a random seed
n_perms = 1  # 00  # number of permutations wanted

# parameters for the default classifier & cross-validation
svm = GridSearchCV(SVC(), {"kernel": ("linear", "rbf"), "C": [0.1, 1, 10]}, cv=5)
cv = ShuffleSplit(nbSplit, test_size=0.2, random_state=21)

# %% parameters to be applied to extract the features
freqbands = {'theta-alpha': [3, 14],  # reboot
             'alpha-beta': [8, 30],  # requested by the reviewer
             'beta-gamma': [14, 40],  # requested by the reviewer
             'theta-alpha-beta': [3, 30],  # requested by the reviewer
             'paper': [3, 40]}

opt_zthresh = [1.4, 1.6, 1.8, 2, 2.2, 2.4, 2.6, 2.8, 3]
opt_val_duration = [2, 3, 4, 5, 6, 7, 8]

opt_trial_duration = [100864 / 256]  # one single trial for everything
fs = 256

test = mat73.loadmat(path_data_root +
    '1_Clinical/Epilepsy_GMD/Data_Epi_MEG_4Classif_concat_NoTrials.mat')
ch_names = test['labels_AAL1']
ch_types = ["eeg" for i in range(np.shape(ch_names)[0])]

# %% test 1 - classification HC vs EP1 - 32 vs 32
grp_id_2use = ['HC', 'EPI 1']
precomp_concat_name = path_data_root + '/concat_epochs_HC_EP1.gz'

perm_idx_filename = path_data_root + '/Permuted_Idx_Classification.gz'
if osp.exists(perm_idx_filename):
    print("Loading existing permuted indices to be applied...")
    with gzip.open(perm_idx_filename, "rb") as file:
        perm = pickle.load(file)
else:
    print("Compute the permuted indices used once for all!")
    perm = rng.permutation(62)  # len(epochs2use) = 62
    with gzip.open(perm_idx_filename, "w") as file:
        pickle.dump([perm], file)
        print('Permuted indices saved')

for f in freqbands:
    precomp_concat_name_f = path_data_root + '/concat_epochs_HC_EP1_' + f + '.gz'
    if osp.exists(precomp_concat_name_f):
        print("Loading existing concatenated precomputations...")
        with gzip.open(precomp_concat_name_f, "rb") as file:
            epochs_concat, labels_concat = pickle.load(file)
    else:
        print("Please consider performing precomputations and/or concatenation of the epochs!")

    fmin = freqbands[f][0]
    fmax = freqbands[f][1]
    results_global = pd.DataFrame()
    results_atm = pd.DataFrame()
    results_ImCoh = pd.DataFrame()
    epochs2use = epochs_concat[f].drop([2, 31, 32, 33, 34],
                                       reason='USER')  # to have the same nb of pz/subjects and because data from the second patient may be corrupted
    for iperm in range(n_perms):  # in case we want to do some permutations to increase the statistical power
        # perm = rng.permutation(len(epochs2use))
        epochs_in_permuted_order = epochs2use[perm[0]]
        data_shuffle = epochs_in_permuted_order.get_data()
        label_shuffle = [labels_concat[f][x] for x in perm[0]]
        nb_trials = len(data_shuffle)  # nb subjects/ pz here
        nb_ROIs = np.shape(data_shuffle)[1]

        ### ImCoh + SVM
        # # here to compute FC on long recordings we increase the time window of interest to speed up the process
        ft = FunctionalTransformer(
            delta=30, ratio=0.5, method="imcoh", fmin=fmin, fmax=fmax
        )
        preproc_meg = Pipeline(steps=[("ft", ft)])  # , ("spd", EnsureSPD())])
        mat_ImCoh = preproc_meg.fit_transform(
            epochs_in_permuted_order)  # actually unfair because does an average of the estimations ie more robust than ATM this way

        ImCoh_cl = np.reshape(mat_ImCoh, (nb_trials, nb_ROIs * nb_ROIs))

        clf_2 = Pipeline([('SVM', svm)])
        # score_ImCoh_SVM = cross_val_score(clf_2, ImCoh_cl, labels, cv=cv, n_jobs=None)
        scores_ImCoh_SVM = cross_validate(clf_2, ImCoh_cl, label_shuffle, cv=cv, n_jobs=None, scoring=scoring,
                                          return_estimator=False)  # returns the estimator objects for each cv split!

        # concatenate ImCoh results in a dedicated dataframe
        pd_ImCoh_SVM = pd.DataFrame.from_dict(scores_ImCoh_SVM)
        temp_results_ImCoh = pd_ImCoh_SVM
        ppl_ImCoh = ["ImCoh+SVM"] * len(pd_ImCoh_SVM)
        temp_results_ImCoh["pipeline"] = ppl_ImCoh
        temp_results_ImCoh["split"] = [nbSplit] * len(ppl_ImCoh)
        temp_results_ImCoh["freq"] = [str(fmin) + '-' + str(fmax)] * len(ppl_ImCoh)

        results_ImCoh = pd.concat((results_ImCoh, temp_results_ImCoh))

        ### ATM + SVM -- attempt to optimize the code to make it faster...
        temp = np.transpose(data_shuffle, (1, 0, 2))
        temp_nc = np.reshape(temp, (np.shape(temp)[0], np.shape(temp)[1] * np.shape(temp)[2]))
        zscored_data = zscore(temp_nc, axis=1)
        # epoching here before computing the avalanches
        temp_zscored_data_ep = np.reshape(zscored_data,
                                          (np.shape(temp)[0], np.shape(temp)[1], np.shape(temp)[2]))
        zscored_data_ep = np.transpose(temp_zscored_data_ep, (1, 0, 2))

        for kk_zthresh in opt_zthresh:
            for kk_val_duration in opt_val_duration:
                ATM = np.empty((nb_trials, nb_ROIs, nb_ROIs))
                for kk_trial in range(nb_trials):
                    list_avalanches, min_size_avalanches, max_size_avalanches, list_avalanches_bysize, temp_ATM = find_avalanches(
                        zscored_data_ep[kk_trial, :, :], thresh=kk_zthresh, val_duration=kk_val_duration)
                    # ATM: nb_trials x nb_ROIs x nb_ROIs matrix
                    ATM[kk_trial, :, :] = temp_ATM

                clf_2 = Pipeline([('SVM', svm)])
                reshape_ATM = np.reshape(ATM, (np.shape(ATM)[0], np.shape(ATM)[1] * np.shape(ATM)[2]))
                fixed_reshape_ATM = np.nan_to_num(reshape_ATM, nan=0)  # replace nan by 0
                temp_ATM = np.nan_to_num(ATM, nan=0)

                # score_ATM_SVM = cross_val_score(clf_2, reshape_ATM, labels, cv=cv, n_jobs=None)
                scores_ATM_SVM = cross_validate(clf_2, fixed_reshape_ATM, label_shuffle, cv=cv, n_jobs=None,
                                                scoring=scoring, return_estimator=False)

                # concatenate ATM results in a dedicated dataframe
                pd_ATM_SVM = pd.DataFrame.from_dict(scores_ATM_SVM)
                temp_results_atm = pd_ATM_SVM
                ppl_atm = ["ATM+SVM"] * len(pd_ATM_SVM)
                temp_results_atm["pipeline"] = ppl_atm
                temp_results_atm["split"] = [nbSplit] * len(ppl_atm)
                temp_results_atm["zthresh"] = [kk_zthresh] * len(ppl_atm)
                temp_results_atm["val_duration"] = [kk_val_duration] * len(ppl_atm)
                temp_results_atm["freq"] = [str(fmin) + '-' + str(fmax)] * len(ppl_atm)

                results_atm = pd.concat((results_atm, temp_results_atm))

    # concatenate results in a single dataframe
    results_global = pd.concat((results_atm, results_ImCoh))
    results_global.to_csv(
        path_csv_root + "/SVM/HC_EP1_IndivOpt_ATM_ImCoh_Comparison_SVM_Classification-allnode-2class-" +
        "-freq-" + str(fmin) + '-' + str(fmax) + '-nbSplit' + str(nbSplit) + ".csv"
    )
    print(
        "saved " +
        path_csv_root + "/SVM/HC_EP1_IndivOpt_ATM_ImCoh_Comparison_SVM_Classification-allnode-2class-" +
        "-freq-" + str(fmin) + '-' + str(fmax) + '-nbSplit' + str(nbSplit) + ".csv"
    )

# %%
results = pd.DataFrame()
freqbands_tot = {'alpha-beta': [8, 30],  # requested by the reviewer
                 'beta-gamma': [14, 40],  # requested by the reviewer
                 'theta_alpha-beta': [3, 30],  # requested by the reviewer
                 'theta_alpha': [3, 14],  # Epilepsy case
                 'paper': [3, 40]}

for f in freqbands_tot:
    fmin = freqbands_tot[f][0]
    fmax = freqbands_tot[f][1]
    temp_results = pd.read_csv(
        path_csv_root + "/SVM/HC_EP1_IndivOpt_ATM_ImCoh_Comparison_SVM_Classification-allnode-2class-" +
        "-freq-" + str(fmin) + '-' + str(fmax) + '-nbSplit' + str(nbSplit) + ".csv"
        )

    results = pd.concat((results, temp_results))

# %%
import matplotlib.pyplot as plt
import seaborn as sns

path_figures_root = "/Users/marieconstance.corsi/Documents/GitHub/Fenicotteri-equilibristi/Figures/Classification/"

results_theta_alpha = results[results["freq"] == '3-14']
results_theta_alpha_ImCoh = results_theta_alpha[results_theta_alpha["pipeline"] == "ImCoh+SVM"]
results_theta_alpha_atm = results_theta_alpha[results_theta_alpha["pipeline"] == "ATM+SVM"]

results_broad = results[results["freq"] == '3-40']
results_broad_ImCoh = results_broad[results_broad["pipeline"] == "ImCoh+SVM"]
results_broad_atm = results_broad[results_broad["pipeline"] == "ATM+SVM"]

results_alpha_beta = results[results["freq"] == '8-30']
results_alpha_beta_ImCoh = results_alpha_beta[results_alpha_beta["pipeline"] == "ImCoh+SVM"]
results_alpha_beta_atm = results_alpha_beta[results_alpha_beta["pipeline"] == "ATM+SVM"]

results_beta_gamma = results[results["freq"] == '14-40']
results_beta_gamma_ImCoh = results_beta_gamma[results_beta_gamma["pipeline"] == "ImCoh+SVM"]
results_beta_gamma_atm = results_beta_gamma[results_beta_gamma["pipeline"] == "ATM+SVM"]

results_theta_alpha_beta = results[results["freq"] == '3-30']
results_theta_alpha_beta_ImCoh = results_theta_alpha_beta[results_theta_alpha_beta["pipeline"] == "ImCoh+SVM"]
results_theta_alpha_beta_atm = results_theta_alpha_beta[results_theta_alpha_beta["pipeline"] == "ATM+SVM"]

# %%
# broad
df_db = pd.DataFrame()
df_res_opt_db = pd.DataFrame()
temp_cfg_opt_db = pd.DataFrame()
df_res_opt_db_median = pd.DataFrame()

# beta-gamma
df_beta_gamma = pd.DataFrame()
df_res_opt_beta_gamma = pd.DataFrame()
temp_cfg_opt_beta_gamma = pd.DataFrame()
df_res_opt_beta_gamma_median = pd.DataFrame()

# theta_alpha
df_theta_alpha = pd.DataFrame()
df_res_opt_theta_alpha = pd.DataFrame()
temp_cfg_opt_theta_alpha = pd.DataFrame()
df_res_opt_theta_alpha_median = pd.DataFrame()

# alpha-beta
df_alpha_beta = pd.DataFrame()
df_res_opt_alpha_beta = pd.DataFrame()
temp_cfg_opt_alpha_beta = pd.DataFrame()
df_res_opt_alpha_beta_median = pd.DataFrame()

# theta_alpha-beta
df_theta_alpha_beta = pd.DataFrame()
df_res_opt_theta_alpha_beta = pd.DataFrame()
temp_cfg_opt_theta_alpha_beta = pd.DataFrame()
df_res_opt_theta_alpha_beta_median = pd.DataFrame()

for kk_zthresh in opt_zthresh:
    for kk_val_duration in opt_val_duration:
        # broad
        temp_data_db = results_broad_atm.loc[(results_broad_atm["zthresh"] == kk_zthresh) &
                                             (results_broad_atm["val_duration"] == kk_val_duration)]
        median_value = temp_data_db["test_accuracy"].median()
        idx_dummy = temp_data_db["test_accuracy"].idxmin()
        temp_df_db = temp_data_db[temp_data_db.index.isin([idx_dummy])]
        temp_df_db.loc[idx_dummy, 'test_accuracy'] = median_value
        df_db = pd.concat((df_db, temp_df_db), ignore_index=True)

        # theta_alpha
        temp_data_theta_alpha = results_theta_alpha_atm.loc[(results_theta_alpha_atm["zthresh"] == kk_zthresh) &
                                                            (results_theta_alpha_atm[
                                                                 "val_duration"] == kk_val_duration)]
        median_value = temp_data_theta_alpha["test_accuracy"].median()
        idx_dummy = temp_data_theta_alpha["test_accuracy"].idxmin()
        temp_df_theta_alpha = temp_data_theta_alpha[temp_data_theta_alpha.index.isin([idx_dummy])]
        temp_df_theta_alpha.loc[idx_dummy, 'test_accuracy'] = median_value
        df_theta_alpha = pd.concat((df_theta_alpha, temp_df_theta_alpha), ignore_index=True)

        # alpha-beta
        temp_data_alpha_beta = results_alpha_beta_atm.loc[(results_alpha_beta_atm["zthresh"] == kk_zthresh) &
                                                          (results_alpha_beta_atm["val_duration"] == kk_val_duration)]
        median_value = temp_data_alpha_beta["test_accuracy"].median()
        idx_dummy = temp_data_alpha_beta["test_accuracy"].idxmin()
        temp_df_alpha_beta = temp_data_alpha_beta[temp_data_alpha_beta.index.isin([idx_dummy])]
        temp_df_alpha_beta.loc[idx_dummy, 'test_accuracy'] = median_value
        df_alpha_beta = pd.concat((df_alpha_beta, temp_df_alpha_beta), ignore_index=True)

        # beta-gamma
        temp_data_beta_gamma = results_beta_gamma_atm.loc[(results_beta_gamma_atm["zthresh"] == kk_zthresh) &
                                                          (results_beta_gamma_atm["val_duration"] == kk_val_duration)]
        median_value = temp_data_beta_gamma["test_accuracy"].median()
        idx_dummy = temp_data_beta_gamma["test_accuracy"].idxmin()
        temp_df_beta_gamma = temp_data_beta_gamma[temp_data_beta_gamma.index.isin([idx_dummy])]
        temp_df_beta_gamma.loc[idx_dummy, 'test_accuracy'] = median_value
        df_beta_gamma = pd.concat((df_beta_gamma, temp_df_beta_gamma), ignore_index=True)

        # theta_alpha_beta
        temp_data_theta_alpha_beta = results_theta_alpha_beta_atm.loc[
            (results_theta_alpha_beta_atm["zthresh"] == kk_zthresh) &
            (results_theta_alpha_beta_atm["val_duration"] == kk_val_duration)]
        median_value = temp_data_theta_alpha_beta["test_accuracy"].median()
        idx_dummy = temp_data_theta_alpha_beta["test_accuracy"].idxmin()
        temp_df_theta_alpha_beta = temp_data_theta_alpha_beta[temp_data_theta_alpha_beta.index.isin([idx_dummy])]
        temp_df_theta_alpha_beta.loc[idx_dummy, 'test_accuracy'] = median_value
        df_theta_alpha_beta = pd.concat((df_theta_alpha_beta, temp_df_theta_alpha_beta), ignore_index=True)

# %%
# broad
max_score_db = df_db["test_accuracy"].max()
idx_max_score_db = df_db["test_accuracy"].idxmax()
temp_cfg_opt_db = df_db[df_db.index.isin([idx_max_score_db])]
df_res_opt_db = pd.concat((df_res_opt_db, temp_cfg_opt_db))

idx_dummy = temp_cfg_opt_db["test_accuracy"].idxmin()
median_db = temp_cfg_opt_db[temp_cfg_opt_db.index.isin([idx_dummy])]
median_db.loc[idx_dummy, 'test_accuracy'] = temp_cfg_opt_db["test_accuracy"].median()
df_res_opt_db_median = pd.concat((df_res_opt_db_median, median_db))

# theta-alpha
max_score_theta_alpha = df_theta_alpha["test_accuracy"].max()
idx_max_score_theta_alpha = df_theta_alpha["test_accuracy"].idxmax()
temp_cfg_opt_theta_alpha = df_theta_alpha[df_theta_alpha.index.isin([idx_max_score_theta_alpha])]
df_res_opt_theta_alpha = pd.concat((df_res_opt_theta_alpha, temp_cfg_opt_theta_alpha))

idx_dummy = temp_cfg_opt_theta_alpha["test_accuracy"].idxmin()
median_theta_alpha = temp_cfg_opt_theta_alpha[temp_cfg_opt_theta_alpha.index.isin([idx_dummy])]
median_theta_alpha.loc[idx_dummy, 'test_accuracy'] = temp_cfg_opt_theta_alpha["test_accuracy"].median()
df_res_opt_theta_alpha_median = pd.concat((df_res_opt_theta_alpha_median, median_theta_alpha))

# alpha-beta
max_score_alpha_beta = df_alpha_beta["test_accuracy"].max()
idx_max_score_alpha_beta = df_alpha_beta["test_accuracy"].idxmax()
temp_cfg_opt_alpha_beta = df_alpha_beta[df_alpha_beta.index.isin([idx_max_score_alpha_beta])]
df_res_opt_alpha_beta = pd.concat((df_res_opt_alpha_beta, temp_cfg_opt_alpha_beta))

idx_dummy = temp_cfg_opt_alpha_beta["test_accuracy"].idxmin()
median_alpha_beta = temp_cfg_opt_alpha_beta[temp_cfg_opt_alpha_beta.index.isin([idx_dummy])]
median_alpha_beta.loc[idx_dummy, 'test_accuracy'] = temp_cfg_opt_alpha_beta["test_accuracy"].median()
df_res_opt_alpha_beta_median = pd.concat((df_res_opt_alpha_beta_median, median_alpha_beta))

# beta-gamma
max_score_beta_gamma = df_beta_gamma["test_accuracy"].max()
idx_max_score_beta_gamma = df_beta_gamma["test_accuracy"].idxmax()
temp_cfg_opt_beta_gamma = df_beta_gamma[df_beta_gamma.index.isin([idx_max_score_beta_gamma])]
df_res_opt_beta_gamma = pd.concat((df_res_opt_beta_gamma, temp_cfg_opt_beta_gamma))

idx_dummy = temp_cfg_opt_beta_gamma["test_accuracy"].idxmin()
median_beta_gamma = temp_cfg_opt_beta_gamma[temp_cfg_opt_beta_gamma.index.isin([idx_dummy])]
median_beta_gamma.loc[idx_dummy, 'test_accuracy'] = temp_cfg_opt_beta_gamma["test_accuracy"].median()
df_res_opt_beta_gamma_median = pd.concat((df_res_opt_beta_gamma_median, median_beta_gamma))

# theta-alpha-beta
max_score_theta_alpha_beta = df_theta_alpha_beta["test_accuracy"].max()
idx_max_score_theta_alpha_beta = df_theta_alpha_beta["test_accuracy"].idxmax()
temp_cfg_opt_theta_alpha_beta = df_theta_alpha_beta[df_theta_alpha_beta.index.isin([idx_max_score_theta_alpha_beta])]
df_res_opt_theta_alpha_beta = pd.concat((df_res_opt_theta_alpha_beta, temp_cfg_opt_theta_alpha_beta))

idx_dummy = temp_cfg_opt_theta_alpha_beta["test_accuracy"].idxmin()
median_theta_alpha_beta = temp_cfg_opt_theta_alpha_beta[temp_cfg_opt_theta_alpha_beta.index.isin([idx_dummy])]
median_theta_alpha_beta.loc[idx_dummy, 'test_accuracy'] = temp_cfg_opt_theta_alpha_beta["test_accuracy"].median()
df_res_opt_theta_alpha_beta_median = pd.concat((df_res_opt_theta_alpha_beta_median, median_theta_alpha_beta))

df_res_opt_db.to_csv(
    path_csv_root + "/SVM/OptConfig_HC_EP1_MEG_ATM_SVM_Classification-allnode_rest_BroadBand.csv"
)
df_res_opt_db_median.to_csv(
    path_csv_root + "/SVM/Median_OptConfig_HC_EP1_MEG_ATM_SVM_Classification-allnode_rest_BroadBand.csv"
)

df_res_opt_theta_alpha.to_csv(
    path_csv_root + "/SVM/OptConfig_HC_EP1_MEG_ATM_SVM_Classification-allnode_rest_theta_alpha_Band.csv"
)
df_res_opt_theta_alpha_median.to_csv(
    path_csv_root + "/SVM/Median_OptConfig_HC_EP1_MEG_ATM_SVM_Classification-allnode_rest_theta_alpha_Band.csv"
)

df_res_opt_alpha_beta.to_csv(
    path_csv_root + "/SVM/OptConfig_HC_EP1_MEG_ATM_SVM_Classification-allnode_rest_alpha_beta_Band.csv"
)
df_res_opt_alpha_beta_median.to_csv(
    path_csv_root + "/SVM/Median_OptConfig_HC_EP1_MEG_ATM_SVM_Classification-allnode_rest_alpha_beta_Band.csv"
)

df_res_opt_beta_gamma.to_csv(
    path_csv_root + "/SVM/OptConfig_HC_EP1_MEG_ATM_SVM_Classification-allnode_rest_beta_gamma_Band.csv"
)
df_res_opt_beta_gamma_median.to_csv(
    path_csv_root + "/SVM/Median_OptConfig_HC_EP1_MEG_ATM_SVM_Classification-allnode_rest_beta_gamma_Band.csv"
)

df_res_opt_theta_alpha_beta.to_csv(
    path_csv_root + "/SVM/OptConfig_HC_EP1_MEG_ATM_SVM_Classification-allnode_rest_df_res_opt_theta_alpha_beta_Band.csv"
)
df_res_opt_theta_alpha_beta_median.to_csv(
    path_csv_root + "/SVM/Median_OptConfig_HC_EP1_MEG_ATM_SVM_Classification-allnode_rest_theta_alpha_beta_Band.csv"
)

# %% best config - aqui TODO - check optimal configuration
# plot
results_broad_opt = results_broad_atm[(results_broad_atm["zthresh"] == float(df_res_opt_db["zthresh"])) & (
            results_broad_atm["val_duration"] == float(df_res_opt_db["val_duration"]))]
results_theta_alpha_opt = results_theta_alpha_atm[
    (results_theta_alpha_atm["zthresh"] == float(df_res_opt_theta_alpha["zthresh"])) & (
                results_theta_alpha_atm["val_duration"] == float(df_res_opt_theta_alpha["val_duration"]))]
results_alpha_beta_opt = results_alpha_beta_atm[
    (results_alpha_beta_atm["zthresh"] == float(df_res_opt_alpha_beta["zthresh"])) & (
                results_alpha_beta_atm["val_duration"] == float(df_res_opt_alpha_beta["val_duration"]))]
results_beta_gamma_opt = results_beta_gamma_atm[
    (results_beta_gamma_atm["zthresh"] == float(df_res_opt_beta_gamma["zthresh"])) & (
                results_beta_gamma_atm["val_duration"] == float(df_res_opt_beta_gamma["val_duration"]))]
results_theta_alpha_beta_opt = results_theta_alpha_beta_atm[
    (results_theta_alpha_beta_atm["zthresh"] == float(df_res_opt_theta_alpha_beta["zthresh"])) & (
                results_theta_alpha_beta_atm["val_duration"] == float(df_res_opt_theta_alpha_beta["val_duration"]))]

results_opt_global = pd.concat((results_broad_opt, results_theta_alpha_opt, results_alpha_beta_opt,
                                results_beta_gamma_opt, results_theta_alpha_beta_opt,
                                results_broad_ImCoh, results_theta_alpha_ImCoh, results_alpha_beta_ImCoh,
                                results_beta_gamma_ImCoh, results_theta_alpha_beta_ImCoh))
# plt.style.use("dark_background")
g = sns.catplot(y="test_accuracy",
                x='pipeline',
                hue="pipeline",
                kind='swarm',  # swarm
                col="freq",
                col_wrap=2,
                # dodge=True,
                # row='zthresh',
                height=4, aspect=4,
                data=results_opt_global)

plt.savefig(path_figures_root + "Opt_HC_EP1_ATM_IMCoh_SVM_Classification_2class-nbSplits" + str(
    nbSplit) + "_broad_MEG.pdf", dpi=300)

# %% Plot répartition des performances médianes (sur les splits) pour chaque configuration testée et  pour chaque bande de fréquence - ATM
sns.set_theme(style='ticks')
plt.style.use("classic")
# broad band
sns.jointplot(data=df_db.round(decimals=2), x='val_duration', y='zthresh', hue="test_accuracy", kind='scatter',
              palette='viridis').fig.suptitle('ATM - Broad band')
plt.savefig(
    path_figures_root + "OptimalConfig_ATM_BroadBand_Edges_HC_EP1_ATM_SVM_Classification_2class_nbSplits" + str(
        nbSplit) + "_MEG.pdf", dpi=300)

# theta-alpha
sns.jointplot(data=df_theta_alpha.round(decimals=2), x='val_duration', y='zthresh', hue="test_accuracy", kind='scatter',
              palette='viridis').fig.suptitle('ATM - Theta-Alpha band')
plt.savefig(
    path_figures_root + "OptimalConfig_ATM_Theta_Alpha_Band_Edges_HC_EP1_ATM_SVM_Classification_2class_nbSplits" + str(
        nbSplit) + "_MEG.pdf", dpi=300)

# alpha-beta
sns.jointplot(data=df_alpha_beta.round(decimals=2), x='val_duration', y='zthresh', hue="test_accuracy", kind='scatter',
              palette='viridis').fig.suptitle('ATM - Alpha-Beta band')
plt.savefig(
    path_figures_root + "OptimalConfig_ATM_Alpha_Beta_Band_Edges_HC_EP1_ATM_SVM_Classification_2class_nbSplits" + str(
        nbSplit) + "_MEG.pdf", dpi=300)

# beta-gamma
sns.jointplot(data=df_beta_gamma.round(decimals=2), x='val_duration', y='zthresh', hue="test_accuracy", kind='scatter',
              palette='viridis').fig.suptitle('ATM - Beta-Gamma band')
plt.savefig(
    path_figures_root + "OptimalConfig_ATM_Beta_Gamma_Band_Edges_HC_EP1_ATM_SVM_Classification_2class_nbSplits" + str(
        nbSplit) + "_MEG.pdf", dpi=300)

# theta-alpha_beta
sns.jointplot(data=df_theta_alpha_beta.round(decimals=2), x='val_duration', y='zthresh', hue="test_accuracy",
              kind='scatter',
              palette='viridis').fig.suptitle('ATM - Theta-Alpha-Beta band')
plt.savefig(
    path_figures_root + "OptimalConfig_ATM_Theta_Alpha_Beta_Band_Edges_HC_EP1_ATM_SVM_Classification_2class_nbSplits" + str(
        nbSplit) + "_MEG.pdf", dpi=300)