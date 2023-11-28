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
freqbands = {'theta': [3, 8],
             'alpha': [8, 14],
             'theta-alpha': [3, 14],  # Epilepsy case
             'paper': [3, 40]}

#opt_zthresh = [2]  # 1.4
#opt_val_duration = [2]
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

#%% test 1 - classification HC vs EP1 - 32 vs 32
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
    results_global = pd.DataFrame()
    results_atm = pd.DataFrame()
    results_plv = pd.DataFrame()
    epochs2use=epochs_concat[f].drop([2, 31, 32, 33, 34], reason='USER') # to have the same nb of pz/subjects and because data from the second patient may be corrupted
    for iperm in range(n_perms): # in case we want to do some permutations to increase the statistical power
        perm = rng.permutation(len(epochs2use))
        epochs_in_permuted_order = epochs2use[perm]
        data_shuffle = epochs_in_permuted_order.get_data()
        label_shuffle = [labels_concat[f][x] for x in perm]
        nb_trials = len(data_shuffle)  # nb subjects/ pz here
        nb_ROIs = np.shape(data_shuffle)[1]

        ### PLV + SVM
        # # here to compute FC on long recordings we increase the time window of interest to speed up the process
        ft = FunctionalTransformer(
            delta=30, ratio=0.5, method="plv", fmin=fmin, fmax=fmax
        )
        preproc_meg = Pipeline(steps=[("ft", ft)])  # , ("spd", EnsureSPD())])
        mat_PLV = preproc_meg.fit_transform(epochs_in_permuted_order) # actually unfair because does an average of the estimations ie more robust than ATM this way

        PLV_cl = np.reshape(mat_PLV, (nb_trials, nb_ROIs * nb_ROIs))
        PLV_nodal_cl = np.sum(mat_PLV,1)

        clf_2 = Pipeline([('SVM', svm)])
        # score_PLV_SVM = cross_val_score(clf_2, PLV_cl, labels, cv=cv, n_jobs=None)
        scores_PLV_SVM = cross_validate(clf_2, PLV_cl,  label_shuffle, cv=cv, n_jobs=None, scoring=scoring, return_estimator=False) # returns the estimator objects for each cv split!
        scores_PLV_nodal_SVM = cross_validate(clf_2, PLV_nodal_cl,  label_shuffle, cv=cv, n_jobs=None, scoring=scoring, return_estimator=False) # returns the estimator objects for each cv split!

        # concatenate PLV results in a dedicated dataframe
        pd_PLV_SVM = pd.DataFrame.from_dict(scores_PLV_SVM)
        temp_results_plv = pd_PLV_SVM
        ppl_plv = ["PLV+SVM"] * len(pd_PLV_SVM)
        temp_results_plv["pipeline"] = ppl_plv
        temp_results_plv["split"] = [nbSplit] * len(ppl_plv)
        temp_results_plv["freq"] = [str(fmin) + '-' + str(fmax)] * len(ppl_plv)

        pd_PLV_nodal_SVM = pd.DataFrame.from_dict(scores_PLV_nodal_SVM)
        temp_results_plv_nodal = pd_PLV_nodal_SVM
        ppl_plv_nodal = ["PLV+SVM-nodal"] * len(pd_PLV_nodal_SVM)
        temp_results_plv_nodal["pipeline"] = ppl_plv_nodal
        temp_results_plv_nodal["split"] = [nbSplit] * len(ppl_plv_nodal)
        temp_results_plv_nodal["freq"] = [str(fmin) + '-' + str(fmax)] * len(ppl_plv_nodal)

        results_plv = pd.concat((results_plv, temp_results_plv, temp_results_plv_nodal))


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
                ATM_nodal = np.sum(temp_ATM, 1)

                #score_ATM_SVM = cross_val_score(clf_2, reshape_ATM, labels, cv=cv, n_jobs=None)
                scores_ATM_SVM = cross_validate(clf_2, fixed_reshape_ATM, label_shuffle, cv=cv, n_jobs=None, scoring=scoring, return_estimator=False)
                scores_ATM_SVM_nodal = cross_validate(clf_2, ATM_nodal, label_shuffle, cv=cv, n_jobs=None,
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

                pd_ATM_nodal_SVM = pd.DataFrame.from_dict(scores_ATM_SVM_nodal)
                temp_results_atm_nodal = pd_ATM_nodal_SVM
                ppl_atm_nodal = ["ATM+SVM-nodal"] * len(pd_ATM_nodal_SVM)
                temp_results_atm_nodal["pipeline"] = ppl_atm_nodal
                temp_results_atm_nodal["split"] = [nbSplit] * len(ppl_atm_nodal)
                temp_results_atm_nodal["zthresh"] = [kk_zthresh] * len(ppl_atm_nodal)
                temp_results_atm_nodal["val_duration"] = [kk_val_duration] * len(ppl_atm_nodal)
                temp_results_atm_nodal["freq"] = [str(fmin) + '-' + str(fmax)] * len(ppl_atm_nodal)

                results_atm = pd.concat((results_atm, temp_results_atm, temp_results_atm_nodal))

    # concatenate results in a single dataframe
    results_global = pd.concat((results_atm, results_plv))
    results_global.to_csv(
            path_csv_root + "/SVM/HC_EP1_IndivOpt_ATM_PLV_Comparison_SVM_Classification-allnode-2class-" +
            "-freq-" + str(fmin) + '-' + str(fmax) + '-nbSplit' + str(nbSplit) + ".csv"
        )
    print(
            "saved " +
            path_csv_root + "/SVM/HC_EP1_IndivOpt_ATM_PLV_Comparison_SVM_Classification-allnode-2class-" +
            "-freq-" + str(fmin) + '-' + str(fmax) + '-nbSplit' + str(nbSplit) + ".csv"
        )


#%%
results=pd.DataFrame()
freqbands_tot = {'theta': [3, 8],
                 'alpha': [8, 14],
                 'theta-alpha': [3, 14],  # Epilepsy case
                 'paper': [3, 40]}

for f in freqbands_tot:
    fmin = freqbands_tot[f][0]
    fmax = freqbands_tot[f][1]
    temp_results = pd.read_csv(         path_csv_root + "/SVM/HC_EP1_IndivOpt_ATM_PLV_Comparison_SVM_Classification-allnode-2class-" +
            "-freq-" + str(fmin) + '-' + str(fmax) + '-nbSplit' + str(nbSplit) + ".csv"
             )

    results = pd.concat((results, temp_results))


#%%
import matplotlib.pyplot as plt
import seaborn as sns
path_figures_root = "/Users/marieconstance.corsi/Documents/GitHub/Fenicotteri-equilibristi/Figures/Classification/"

results_theta_alpha=results[results["freq"]=='3-14']
results_theta_alpha_plv=results_theta_alpha[results_theta_alpha["pipeline"]=="PLV+SVM"]
results_theta_alpha_plv_nodal=results_theta_alpha[results_theta_alpha["pipeline"]=="PLV+SVM-nodal"]
results_theta_alpha_atm=results_theta_alpha[results_theta_alpha["pipeline"]=="ATM+SVM"]
results_theta_alpha_atm_nodal=results_theta_alpha[results_theta_alpha["pipeline"]=="ATM+SVM-nodal"]

results_theta=results[results["freq"]=='3-8']
results_theta_plv=results_theta[results_theta["pipeline"]=="PLV+SVM"]
results_theta_plv_nodal=results_theta[results_theta["pipeline"]=="PLV+SVM-nodal"]
results_theta_atm=results_theta[results_theta["pipeline"]=="ATM+SVM"]
results_theta_atm_nodal=results_theta[results_theta["pipeline"]=="ATM+SVM-nodal"]

results_alpha=results[results["freq"]=='8-14']
results_alpha_plv=results_alpha[results_alpha["pipeline"]=="PLV+SVM"]
results_alpha_plv_nodal=results_alpha[results_alpha["pipeline"]=="PLV+SVM-nodal"]
results_alpha_atm=results_alpha[results_alpha["pipeline"]=="ATM+SVM"]
results_alpha_atm_nodal=results_alpha[results_alpha["pipeline"]=="ATM+SVM-nodal"]

results_broad=results[results["freq"]=='3-40']
results_broad_plv=results_broad[results_broad["pipeline"]=="PLV+SVM"]
results_broad_plv_nodal=results_broad[results_broad["pipeline"]=="PLV+SVM-nodal"]
results_broad_atm=results_broad[results_broad["pipeline"]=="ATM+SVM"]
results_broad_atm_nodal=results_broad[results_broad["pipeline"]=="ATM+SVM-nodal"]

#%%
df_res_opt_db = pd.DataFrame()
df_res_opt_db_nodal = pd.DataFrame()
temp_cfg_opt_db = pd.DataFrame()
temp_cfg_opt_db_nodal = pd.DataFrame()
df_db_nodal = pd.DataFrame()
df_res_opt_theta_alpha = pd.DataFrame()
temp_cfg_opt_theta_alpha = pd.DataFrame()
df_res_opt_theta_alpha_nodal = pd.DataFrame()
temp_cfg_opt_theta_alpha_nodal = pd.DataFrame()
df_res_opt_theta = pd.DataFrame()
temp_cfg_opt_theta = pd.DataFrame()
df_res_opt_theta_nodal = pd.DataFrame()
temp_cfg_opt_theta_nodal = pd.DataFrame()
df_res_opt_alpha = pd.DataFrame()
temp_cfg_opt_alpha = pd.DataFrame()
df_res_opt_alpha_nodal = pd.DataFrame()
temp_cfg_opt_alpha_nodal = pd.DataFrame()

df_db = pd.DataFrame()
df_theta_alpha = pd.DataFrame()
df_theta = pd.DataFrame()
df_alpha = pd.DataFrame()
df_db_nodal = pd.DataFrame()
df_theta_alpha_nodal = pd.DataFrame()
df_theta_nodal = pd.DataFrame()
df_alpha_nodal = pd.DataFrame()

df_res_opt_theta_alpha_median = pd.DataFrame()
df_res_opt_theta_median = pd.DataFrame()
df_res_opt_alpha_median = pd.DataFrame()
df_res_opt_db_median = pd.DataFrame()

df_res_opt_theta_alpha_median_nodal = pd.DataFrame()
df_res_opt_theta_median_nodal = pd.DataFrame()
df_res_opt_alpha_median_nodal = pd.DataFrame()
df_res_opt_db_median_nodal = pd.DataFrame()

for kk_zthresh in opt_zthresh:
    for kk_val_duration in opt_val_duration:
        temp_data_db = results_broad_atm.loc[(results_broad_atm["zthresh"]==kk_zthresh) &
                                         (results_broad_atm["val_duration"]==kk_val_duration)]
        median_value=temp_data_db["test_accuracy"].median()

        idx_dummy = temp_data_db["test_accuracy"].idxmin()
        temp_df_db = temp_data_db[temp_data_db.index.isin([idx_dummy])]
        temp_df_db.loc[idx_dummy, 'test_accuracy'] = median_value

        df_db = pd.concat((df_db, temp_df_db), ignore_index=True)

        #nodal
        temp_data_db_nodal = results_broad_atm_nodal.loc[(results_broad_atm_nodal["zthresh"]==kk_zthresh) &
                                         (results_broad_atm_nodal["val_duration"]==kk_val_duration)]
        median_value=temp_data_db_nodal["test_accuracy"].median()

        idx_dummy = temp_data_db_nodal["test_accuracy"].idxmin()
        temp_df_db_nodal = temp_data_db_nodal[temp_data_db_nodal.index.isin([idx_dummy])]
        temp_df_db_nodal.loc[idx_dummy, 'test_accuracy'] = median_value

        df_db_nodal = pd.concat((df_db_nodal, temp_df_db_nodal), ignore_index=True)



        temp_data_theta_alpha = results_theta_alpha_atm.loc[(results_theta_alpha_atm["zthresh"]==kk_zthresh) &
                                       (results_theta_alpha_atm["val_duration"]==kk_val_duration)]
        median_value=temp_data_theta_alpha["test_accuracy"].median()

        idx_dummy = temp_data_theta_alpha["test_accuracy"].idxmin()
        temp_df_theta_alpha = temp_data_theta_alpha[temp_data_theta_alpha.index.isin([idx_dummy])]
        temp_df_theta_alpha.loc[idx_dummy, 'test_accuracy'] = median_value

        df_theta_alpha = pd.concat((df_theta_alpha, temp_df_theta_alpha), ignore_index=True)

        # nodal
        temp_data_theta_alpha_nodal = results_theta_alpha_atm_nodal.loc[(results_theta_alpha_atm_nodal["zthresh"]==kk_zthresh) &
                                       (results_theta_alpha_atm_nodal["val_duration"]==kk_val_duration)]
        median_value=temp_data_theta_alpha_nodal["test_accuracy"].median()

        idx_dummy = temp_data_theta_alpha_nodal["test_accuracy"].idxmin()
        temp_df_theta_alpha_nodal = temp_data_theta_alpha_nodal[temp_data_theta_alpha_nodal.index.isin([idx_dummy])]
        temp_df_theta_alpha_nodal.loc[idx_dummy, 'test_accuracy'] = median_value

        df_theta_alpha_nodal = pd.concat((df_theta_alpha_nodal, temp_df_theta_alpha_nodal), ignore_index=True)

        temp_data_theta = results_theta_atm.loc[(results_theta_atm["zthresh"]==kk_zthresh) &
                                       (results_theta_atm["val_duration"]==kk_val_duration)]
        median_value=temp_data_theta["test_accuracy"].median()

        idx_dummy = temp_data_theta["test_accuracy"].idxmin()
        temp_df_theta = temp_data_theta[temp_data_theta.index.isin([idx_dummy])]
        temp_df_theta.loc[idx_dummy, 'test_accuracy'] = median_value

        df_theta = pd.concat((df_theta, temp_df_theta), ignore_index=True)

        # nodal
        temp_data_theta_nodal = results_theta_atm_nodal.loc[(results_theta_atm_nodal["zthresh"]==kk_zthresh) &
                                       (results_theta_atm_nodal["val_duration"]==kk_val_duration)]
        median_value=temp_data_theta_nodal["test_accuracy"].median()

        idx_dummy = temp_data_theta_nodal["test_accuracy"].idxmin()
        temp_df_theta_nodal = temp_data_theta_nodal[temp_data_theta_nodal.index.isin([idx_dummy])]
        temp_df_theta_nodal.loc[idx_dummy, 'test_accuracy'] = median_value

        df_theta_nodal = pd.concat((df_theta_nodal, temp_df_theta_nodal), ignore_index=True)

        temp_data_alpha = results_alpha_atm.loc[(results_alpha_atm["zthresh"]==kk_zthresh) &
                                       (results_alpha_atm["val_duration"]==kk_val_duration)]
        median_value=temp_data_alpha["test_accuracy"].median()

        idx_dummy = temp_data_alpha["test_accuracy"].idxmin()
        temp_df_alpha = temp_data_alpha[temp_data_alpha.index.isin([idx_dummy])]
        temp_df_alpha.loc[idx_dummy, 'test_accuracy'] = median_value

        df_alpha = pd.concat((df_alpha, temp_df_alpha), ignore_index=True)

        # nodal
        temp_data_alpha_nodal = results_alpha_atm_nodal.loc[(results_alpha_atm_nodal["zthresh"]==kk_zthresh) &
                                       (results_alpha_atm_nodal["val_duration"]==kk_val_duration)]
        median_value=temp_data_alpha_nodal["test_accuracy"].median()

        idx_dummy = temp_data_alpha_nodal["test_accuracy"].idxmin()
        temp_df_alpha_nodal = temp_data_alpha_nodal[temp_data_alpha_nodal.index.isin([idx_dummy])]
        temp_df_alpha_nodal.loc[idx_dummy, 'test_accuracy'] = median_value

        df_alpha_nodal = pd.concat((df_alpha_nodal, temp_df_alpha_nodal), ignore_index=True)


max_score_db=df_db["test_accuracy"].max()
idx_max_score_db=df_db["test_accuracy"].idxmax()
temp_cfg_opt_db = df_db[df_db.index.isin([idx_max_score_db])]
df_res_opt_db = pd.concat((df_res_opt_db, temp_cfg_opt_db))

max_score_theta_alpha=df_theta_alpha["test_accuracy"].max()
idx_max_score_theta_alpha=df_theta_alpha["test_accuracy"].idxmax()
temp_cfg_opt_theta_alpha = df_theta_alpha[df_theta_alpha.index.isin([idx_max_score_theta_alpha])]
df_res_opt_theta_alpha = pd.concat((df_res_opt_theta_alpha, temp_cfg_opt_theta_alpha))

max_score_theta=df_theta["test_accuracy"].max()
idx_max_score_theta=df_theta["test_accuracy"].idxmax()
temp_cfg_opt_theta = df_theta[df_theta.index.isin([idx_max_score_theta])]
df_res_opt_theta = pd.concat((df_res_opt_theta, temp_cfg_opt_theta))

max_score_alpha=df_alpha["test_accuracy"].max()
idx_max_score_alpha=df_alpha["test_accuracy"].idxmax()
temp_cfg_opt_alpha = df_alpha[df_alpha.index.isin([idx_max_score_alpha])]
df_res_opt_alpha = pd.concat((df_res_opt_alpha, temp_cfg_opt_alpha))

max_score_db_nodal=df_db_nodal["test_accuracy"].max()
idx_max_score_db_nodal=df_db_nodal["test_accuracy"].idxmax()
temp_cfg_opt_db_nodal = df_db_nodal[df_db_nodal.index.isin([idx_max_score_db_nodal])]
df_res_opt_db_nodal = pd.concat((df_res_opt_db_nodal, temp_cfg_opt_db_nodal))

max_score_theta_alpha_nodal=df_theta_alpha_nodal["test_accuracy"].max()
idx_max_score_theta_alpha_nodal=df_theta_alpha_nodal["test_accuracy"].idxmax()
temp_cfg_opt_theta_alpha_nodal = df_theta_alpha_nodal[df_theta_alpha_nodal.index.isin([idx_max_score_theta_alpha_nodal])]
df_res_opt_theta_alpha_nodal = pd.concat((df_res_opt_theta_alpha_nodal, temp_cfg_opt_theta_alpha_nodal))

max_score_theta_nodal=df_theta_nodal["test_accuracy"].max()
idx_max_score_theta_nodal=df_theta_nodal["test_accuracy"].idxmax()
temp_cfg_opt_theta_nodal = df_theta_nodal[df_theta_nodal.index.isin([idx_max_score_theta_nodal])]
df_res_opt_theta_nodal = pd.concat((df_res_opt_theta_nodal, temp_cfg_opt_theta_nodal))

max_score_alpha_nodal=df_alpha_nodal["test_accuracy"].max()
idx_max_score_alpha_nodal=df_alpha_nodal["test_accuracy"].idxmax()
temp_cfg_opt_alpha_nodal = df_alpha_nodal[df_alpha_nodal.index.isin([idx_max_score_alpha_nodal])]
df_res_opt_alpha_nodal = pd.concat((df_res_opt_alpha_nodal, temp_cfg_opt_alpha_nodal))

idx_dummy = temp_cfg_opt_theta_alpha["test_accuracy"].idxmin()
median_theta_alpha = temp_cfg_opt_theta_alpha[temp_cfg_opt_theta_alpha.index.isin([idx_dummy])]
median_theta_alpha.loc[idx_dummy, 'test_accuracy'] = temp_cfg_opt_theta_alpha["test_accuracy"].median()
df_res_opt_theta_alpha_median = pd.concat((df_res_opt_theta_alpha_median, median_theta_alpha))

idx_dummy = temp_cfg_opt_theta["test_accuracy"].idxmin()
median_theta = temp_cfg_opt_theta[temp_cfg_opt_theta.index.isin([idx_dummy])]
median_theta.loc[idx_dummy, 'test_accuracy'] = temp_cfg_opt_theta["test_accuracy"].median()
df_res_opt_theta_median = pd.concat((df_res_opt_theta_median, median_theta))

idx_dummy = temp_cfg_opt_alpha["test_accuracy"].idxmin()
median_alpha = temp_cfg_opt_alpha[temp_cfg_opt_alpha.index.isin([idx_dummy])]
median_alpha.loc[idx_dummy, 'test_accuracy'] = temp_cfg_opt_alpha["test_accuracy"].median()
df_res_opt_alpha_median = pd.concat((df_res_opt_alpha_median, median_alpha))

idx_dummy = temp_cfg_opt_db_nodal["test_accuracy"].idxmin()
median_db_nodal = temp_cfg_opt_db_nodal[temp_cfg_opt_db_nodal.index.isin([idx_dummy])]
median_db_nodal.loc[idx_dummy, 'test_accuracy'] = temp_cfg_opt_db_nodal["test_accuracy"].median()
df_res_opt_db_median_nodal = pd.concat((df_res_opt_db_median_nodal, median_db_nodal))

idx_dummy = temp_cfg_opt_theta_alpha_nodal["test_accuracy"].idxmin()
median_theta_alpha_nodal = temp_cfg_opt_theta_alpha_nodal[temp_cfg_opt_theta_alpha_nodal.index.isin([idx_dummy])]
median_theta_alpha_nodal.loc[idx_dummy, 'test_accuracy'] = temp_cfg_opt_theta_alpha_nodal["test_accuracy"].median()
df_res_opt_theta_alpha_median_nodal = pd.concat((df_res_opt_theta_alpha_median_nodal, median_theta_alpha_nodal))

idx_dummy = temp_cfg_opt_theta_nodal["test_accuracy"].idxmin()
median_theta_nodal = temp_cfg_opt_theta_alpha_nodal[temp_cfg_opt_theta_nodal.index.isin([idx_dummy])]
median_theta_nodal.loc[idx_dummy, 'test_accuracy'] = temp_cfg_opt_theta_nodal["test_accuracy"].median()
df_res_opt_theta_median_nodal = pd.concat((df_res_opt_theta_median_nodal, median_theta_nodal))

idx_dummy = temp_cfg_opt_alpha_nodal["test_accuracy"].idxmin()
median_alpha_nodal = temp_cfg_opt_theta_alpha_nodal[temp_cfg_opt_alpha_nodal.index.isin([idx_dummy])]
median_alpha_nodal.loc[idx_dummy, 'test_accuracy'] = temp_cfg_opt_alpha_nodal["test_accuracy"].median()
df_res_opt_alpha_median_nodal = pd.concat((df_res_opt_alpha_median_nodal, median_alpha_nodal))


df_res_opt_db.to_csv(
    path_csv_root + "/SVM/OptConfig_HC_EP1_MEG_ATM_SVM_ClassificationRebuttal-allnode_rest_BroadBand.csv"
)
df_res_opt_db_median.to_csv(
    path_csv_root + "/SVM/Median_OptConfig_HC_EP1_MEG_ATM_SVM_ClassificationRebuttal-allnode_rest_BroadBand.csv"
)
df_res_opt_theta_alpha.to_csv(
    path_csv_root + "/SVM/OptConfig_HC_EP1_MEG_ATM_SVM_ClassificationRebuttal-allnode_rest_theta_alphaBand.csv"
)
df_res_opt_theta_alpha_median.to_csv(
    path_csv_root + "/SVM/Median_OptConfig_HC_EP1_MEG_ATM_SVM_ClassificationRebuttal-allnode_rest_theta_alphaBand.csv"
)

df_res_opt_theta.to_csv(
    path_csv_root + "/SVM/OptConfig_HC_EP1_MEG_ATM_SVM_ClassificationRebuttal-allnode_rest_theta_Band.csv"
)
df_res_opt_theta_median.to_csv(
    path_csv_root + "/SVM/Median_OptConfig_HC_EP1_MEG_ATM_SVM_ClassificationRebuttal-allnode_rest_theta_Band.csv"
)
df_res_opt_alpha.to_csv(
    path_csv_root + "/SVM/OptConfig_HC_EP1_MEG_ATM_SVM_ClassificationRebuttal-allnode_rest_alphaBand.csv"
)
df_res_opt_alpha_median.to_csv(
    path_csv_root + "/SVM/Median_OptConfig_HC_EP1_MEG_ATM_SVM_ClassificationRebuttal-allnode_rest_alphaBand.csv"
)

df_res_opt_db_nodal.to_csv(
    path_csv_root + "/SVM/OptConfig_HC_EP1_MEG_ATM_SVM_ClassificationRebuttal-nodal_rest_BroadBand.csv"
)
df_res_opt_db_median_nodal.to_csv(
    path_csv_root + "/SVM/Median_OptConfig_HC_EP1_MEG_ATM_SVM_ClassificationRebuttal-nodal_rest_BroadBand.csv"
)
df_res_opt_theta_alpha_nodal.to_csv(
    path_csv_root + "/SVM/OptConfig_HC_EP1_MEG_ATM_SVM_ClassificationRebuttal-nodal_rest_theta_alphaBand.csv"
)
df_res_opt_theta_alpha_median_nodal.to_csv(
    path_csv_root + "/SVM/Median_OptConfig_HC_EP1_MEG_ATM_SVM_ClassificationRebuttal-nodal_rest_theta_alphaBand.csv"
)
df_res_opt_theta_nodal.to_csv(
    path_csv_root + "/SVM/OptConfig_HC_EP1_MEG_ATM_SVM_ClassificationRebuttal-nodal_rest_theta_Band.csv"
)
df_res_opt_theta_median_nodal.to_csv(
    path_csv_root + "/SVM/Median_OptConfig_HC_EP1_MEG_ATM_SVM_ClassificationRebuttal-nodal_rest_theta_Band.csv"
)
df_res_opt_alpha_nodal.to_csv(
    path_csv_root + "/SVM/OptConfig_HC_EP1_MEG_ATM_SVM_ClassificationRebuttal-nodal_rest_alphaBand.csv"
)
df_res_opt_alpha_median_nodal.to_csv(
    path_csv_root + "/SVM/Median_OptConfig_HC_EP1_MEG_ATM_SVM_ClassificationRebuttal-nodal_rest_alphaBand.csv"
)


#%% best config :
#  - theta-alpha [z=2.6, val_duration:6] - 0.92 median
#  - broad [z=2, val_duration:7] - 0.77 median

# plot
results_theta_alpha_opt=results_theta_alpha_atm[(results_theta_alpha_atm["zthresh"]==float(df_res_opt_theta_alpha["zthresh"])) & (results_theta_alpha_atm["val_duration"]==float(df_res_opt_theta_alpha["val_duration"]))]
results_broad_opt=results_broad_atm[(results_broad_atm["zthresh"]==float(df_res_opt_db["zthresh"])) & (results_broad_atm["val_duration"]==float(df_res_opt_db["val_duration"]))]
results_theta_opt=results_theta_atm[(results_theta_atm["zthresh"]==float(df_res_opt_theta["zthresh"])) & (results_theta_atm["val_duration"]==float(df_res_opt_theta["val_duration"]))]
results_alpha_opt=results_alpha_atm[(results_alpha_atm["zthresh"]==float(df_res_opt_alpha["zthresh"])) & (results_alpha_atm["val_duration"]==float(df_res_opt_alpha["val_duration"]))]

results_theta_alpha_nodal_opt=results_theta_alpha_atm_nodal[(results_theta_alpha_atm_nodal["zthresh"]==float(df_res_opt_theta_alpha_nodal["zthresh"])) & (results_theta_alpha_atm_nodal["val_duration"]==float(df_res_opt_theta_alpha_nodal["val_duration"]))]
results_broad_nodal_opt=results_broad_atm_nodal[(results_broad_atm_nodal["zthresh"]==float(df_res_opt_db_nodal["zthresh"])) & (results_broad_atm_nodal["val_duration"]==float(df_res_opt_db_nodal["val_duration"]))]
results_theta_nodal_opt=results_theta_atm_nodal[(results_theta_atm_nodal["zthresh"]==float(df_res_opt_theta_nodal["zthresh"])) & (results_theta_atm_nodal["val_duration"]==float(df_res_opt_theta_nodal["val_duration"]))]
results_alpha_nodal_opt=results_alpha_atm_nodal[(results_alpha_atm_nodal["zthresh"]==float(df_res_opt_alpha_nodal["zthresh"])) & (results_alpha_atm_nodal["val_duration"]==float(df_res_opt_alpha_nodal["val_duration"]))]

results_opt_global = pd.concat((results_theta_alpha_opt, results_broad_opt, results_theta_opt, results_alpha_opt,
                                results_theta_alpha_nodal_opt, results_broad_nodal_opt, results_theta_nodal_opt, results_alpha_nodal_opt,
                                results_theta_alpha_plv, results_broad_plv,results_theta_plv, results_alpha_plv,
                                results_theta_alpha_plv_nodal, results_broad_plv_nodal,results_theta_plv_nodal, results_alpha_plv_nodal))
#plt.style.use("dark_background")
g = sns.catplot(y="test_accuracy",
                x='pipeline',
                hue="pipeline",
                kind='swarm',  # swarm
                col="freq",
                col_wrap=2,
                # dodge=True,
                #row='zthresh',
                height=4, aspect=3,
                data=results_opt_global)

plt.savefig(path_figures_root + "Opt_HC_EP1_ATM_vs_PLV_SVM_Classification_2class-nbSplits"+str(nbSplit)+"_broad_MEG.pdf", dpi=300)


#%% Plot répartition des performances médianes (sur les splits) pour chaque configuration testée et  pour chaque bande de fréquence - ATM
sns.set_theme(style='ticks')
plt.style.use("classic")
# broad band
sns.jointplot(data=df_db.round(decimals=2), x='val_duration', y='zthresh', hue ="test_accuracy", kind='scatter',
              palette='viridis').fig.suptitle('ATM - Broad band (3-40Hz)')
plt.savefig(
    path_figures_root + "OptimalConfig_ATM_BroadBand_Edges_HC_EP1_ATM_SVM_Classification_2class_nbSplits" + str(
        nbSplit) + "_MEG.pdf", dpi=300)

sns.jointplot(data=df_db_nodal.round(decimals=2), x='val_duration', y='zthresh', hue ="test_accuracy", kind='scatter',
              palette='viridis').fig.suptitle('ATM nodal - Broad band (3-40Hz)')
plt.savefig(
    path_figures_root + "OptimalConfig_ATM_BroadBand_Nodal_HC_EP1_ATM_SVM_Classification_2class_nbSplits" + str(
        nbSplit) + "_MEG.pdf", dpi=300)

# theta
sns.jointplot(data=df_theta.round(decimals=2), x='val_duration', y='zthresh', hue ="test_accuracy", kind='scatter',
              palette='viridis').fig.suptitle('ATM - Theta band (3-8Hz)')
plt.savefig(
    path_figures_root + "OptimalConfig_ATM_ThetaBand_Edges_HC_EP1_ATM_SVM_Classification_2class_nbSplits" + str(
        nbSplit) + "_MEG.pdf", dpi=300)

sns.jointplot(data=df_theta_nodal.round(decimals=2), x='val_duration', y='zthresh', hue ="test_accuracy", kind='scatter',
              palette='viridis').fig.suptitle('ATM nodal - Theta band (3-8Hz)')
plt.savefig(
    path_figures_root + "OptimalConfig_ATM_ThetaBand_Nodal_HC_EP1_ATM_SVM_Classification_2class_nbSplits" + str(
        nbSplit) + "_MEG.pdf", dpi=300)

# alpha
sns.jointplot(data=df_alpha.round(decimals=2), x='val_duration', y='zthresh', hue ="test_accuracy", kind='scatter',
              palette='viridis').fig.suptitle('ATM - Alpha band (8-14Hz)')
plt.savefig(
    path_figures_root + "OptimalConfig_ATM_AlphaBand_Edges_HC_EP1_ATM_SVM_Classification_2class_nbSplits" + str(
        nbSplit) + "_MEG.pdf", dpi=300)

sns.jointplot(data=df_alpha_nodal.round(decimals=2), x='val_duration', y='zthresh', hue ="test_accuracy", kind='scatter',
              palette='viridis').fig.suptitle('ATM nodal - Alpha band (8-14Hz)')
plt.savefig(
    path_figures_root + "OptimalConfig_ATM_AlphaBand_Nodal_HC_EP1_ATM_SVM_Classification_2class_nbSplits" + str(
        nbSplit) + "_MEG.pdf", dpi=300)

# theta_alpha
sns.jointplot(data=df_theta_alpha.round(decimals=2), x='val_duration', y='zthresh', hue ="test_accuracy", kind='scatter',
              palette='viridis').fig.suptitle('ATM - Theta-Alpha band (3-14Hz)')
plt.savefig(
    path_figures_root + "OptimalConfig_ATM_ThetaAlphaBand_Edges_HC_EP1_ATM_SVM_Classification_2class_nbSplits" + str(
        nbSplit) + "_MEG.pdf", dpi=300)

sns.jointplot(data=df_theta_alpha_nodal.round(decimals=2), x='val_duration', y='zthresh', hue ="test_accuracy", kind='scatter',
              palette='viridis').fig.suptitle('ATM nodal - Theta-Alpha band (3-14Hz)')
plt.savefig(
    path_figures_root + "OptimalConfig_ATM_ThetaAlphaBand_Nodal_HC_EP1_ATM_SVM_Classification_2class_nbSplits" + str(
        nbSplit) + "_MEG.pdf", dpi=300)