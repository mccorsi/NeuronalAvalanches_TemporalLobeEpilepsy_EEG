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
# csp = CSP_MNE(n_components=kk_components, reg=None, log=True, norm_trace=False, cov_est='epoch') # compute 1 cov/epoch and then avg / class
cv = ShuffleSplit(nbSplit, test_size=0.2, random_state=21)
# clf_0 = Pipeline([('CSP', csp), ('SVM', svm)])


# %% parameters to be applied to extract the features
#events = ["HC", "EP1"]#, "EP2", "EP3", "EP4"]

#events_id={"HC": 0, "EP1": 1, "EP2":2, "EP3":3, "EP4": 4}
freqbands = {'theta-alpha': [3, 14],  # Epilepsy case
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

results_atm = pd.DataFrame()
#results_csp = pd.DataFrame()

for f in freqbands:
    fmin = freqbands[f][0]
    fmax = freqbands[f][1]
    epochs2use=epochs_concat[f].drop([32, 33, 34], reason='USER') # to have the same nb of pz/subjects
    for iperm in range(n_perms): # in case we want to do some permutations to increase the statistical power
        perm = rng.permutation(len(epochs2use))
        epochs_in_permuted_order = epochs2use[perm]
        data_shuffle = epochs_in_permuted_order.get_data()
        label_shuffle = [labels_concat[f][x] for x in perm]

        # ### CSP + SVM: classical
        # #score_CSP_SVM = cross_val_score(clf_0, data_shuffle, label_shuffle, cv=cv, n_jobs=None)
        # scores_CSP_SVM = cross_validate(clf_0, data_shuffle, label_shuffle, cv=cv, n_jobs=None, scoring=scoring)
        # pd_CSP_SVM = pd.DataFrame.from_dict(scores_CSP_SVM)
        # # concatenate CSP results in a dedicated dataframe
        # temp_results_csp = pd_CSP_SVM
        # ppl_csp = ["CSP+SVM"] * len(pd_CSP_SVM)
        # temp_results_csp["pipeline"] = ppl_csp
        # temp_results_csp["split"] = [nbSplit] * len(ppl_csp)
        # temp_results_csp["n_csp_comp"] = [kk_components] * len(ppl_csp)
        # temp_results_csp["zthresh"] = [3] * len(ppl_csp) # fake values to remain consistent
        # temp_results_csp["val_duration"] = [8] * len(ppl_csp) # fake values to remain consistent
        # temp_results_csp["freq"] = [str(fmin) + '-' + str(fmax)] * len(ppl_csp)
        #
        # results_csp = pd.concat((results_csp, temp_results_csp))

        ### ATM + SVM -- attempt to optimize the code to make it faster...
        temp = np.transpose(data_shuffle, (1, 0, 2))
        temp_nc = np.reshape(temp, (np.shape(temp)[0], np.shape(temp)[1] * np.shape(temp)[2]))
        zscored_data = zscore(temp_nc, axis=1)
        # epoching here before computing the avalanches
        temp_zscored_data_ep = np.reshape(zscored_data,
                                          (np.shape(temp)[0], np.shape(temp)[1], np.shape(temp)[2]))
        zscored_data_ep = np.transpose(temp_zscored_data_ep, (1, 0, 2))

        nb_trials = len(data_shuffle)  # nb subjects/ pz here
        nb_ROIs = np.shape(data_shuffle)[1]

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
                #score_ATM_SVM = cross_val_score(clf_2, reshape_ATM, labels, cv=cv, n_jobs=None)
                scores_ATM_SVM = cross_validate(clf_2, fixed_reshape_ATM, label_shuffle, cv=cv, n_jobs=None, scoring=scoring)

                # concatenate ATM results in a dedicated dataframe
                pd_ATM_SVM = pd.DataFrame.from_dict(scores_ATM_SVM)
                temp_results_atm = pd_ATM_SVM
                ppl_atm = ["ATM+SVM"] * len(pd_ATM_SVM)
                temp_results_atm["pipeline"] = ppl_atm
                temp_results_atm["split"] = [nbSplit] * len(ppl_atm)
#                temp_results_atm["n_csp_comp"] = [kk_components] * len(ppl_atm)
                temp_results_atm["zthresh"] = [kk_zthresh] * len(ppl_atm)
                temp_results_atm["val_duration"] = [kk_val_duration] * len(ppl_atm)
                temp_results_atm["freq"] = [str(fmin) + '-' + str(fmax)] * len(ppl_atm)
                results_atm = pd.concat((results_atm, temp_results_atm))

    # concatenate results in a single dataframe
    results_global = results_atm # pd.concat((results_atm, results_csp))
    results_global.to_csv(
            path_csv_root + "/SVM/HC_EP1_IndivOpt_ATM_CSP_PLV_Comparison_SVM_Classification-allnode-2class-" + #"n_csp_cmp-" + str(
             #   kk_components) +
            "-freq-" + str(fmin) + '-' + str(fmax) + '-nbSplit' + str(nbSplit) + ".csv"
        )
    print(
            "saved " +
            path_csv_root + "/SVM/HC_EP1_IndivOpt_ATM_CSP_PLV_Comparison_SVM_Classification-allnode-2class-" + "n_csp_cmp-" +
               # str(kk_components) +
            "-freq-" + str(fmin) + '-' + str(fmax) + '-nbSplit' + str(nbSplit) + ".csv"
        )


# cf https://mne.tools/dev/auto_tutorials/raw/20_event_arrays.html

#%%
freqbands_temp = {'theta-alpha': [3, 14]}#,  # Epilepsy case
             #'paper': [3, 40]}
results=pd.DataFrame()
for f in freqbands:
    fmin = freqbands[f][0]
    fmax = freqbands[f][1]
    temp_results = pd.read_csv(path_csv_root + "/SVM/HC_EP1_IndivOpt_ATM_CSP_PLV_Comparison_SVM_Classification-allnode-2class-" +
                "-freq-" + str(fmin) + '-' + str(fmax) + '-nbSplit' + str(nbSplit) + ".csv"
            )
    results = pd.concat((results, temp_results))

#%%
import matplotlib.pyplot as plt
import seaborn as sns
path_figures_root = "/Users/marieconstance.corsi/Documents/GitHub/Fenicotteri-equilibristi/Figures/Classification/"

results_theta_alpha=results[results["freq"]=='3-14']
plt.style.use("dark_background")
g = sns.catplot(y="test_accuracy",
                x='val_duration',
                hue="val_duration",
                kind="swarm",
                #bw=.25, cut=0, #split=True,
                #dodge=True,
                col='zthresh',
                col_wrap=3,
                #s=0.75, alpha=.9,
                height=3, aspect=5,
                s=2.1, #alpha=.75,
                data=results_theta_alpha)

plt.savefig(path_figures_root + "PrelimTests_HC_EP1_IndivOpt_ATM_CSP_PLV_Comparison_SVM_Classification_allnode-2class-nbSplits"+str(nbSplit)+"_theta_alpha_MEG.pdf", dpi=300)

results_broad=results[results["freq"]=='3-40']
plt.style.use("dark_background")
g = sns.catplot(y="test_accuracy",
                x='val_duration',
                hue="val_duration",
                kind="swarm",
                #bw=.25, cut=0, #split=True,
                #dodge=True,
                col='zthresh',
                col_wrap=3,
                #s=0.75, alpha=.9,
                height=3, aspect=5,
                s=2.1, #alpha=.75,
                data=results_broad)

plt.savefig(path_figures_root + "PrelimTests_HC_EP1_IndivOpt_ATM_CSP_PLV_Comparison_SVM_Classification_allnode-2class-nbSplits"+str(nbSplit)+"_broad_MEG.pdf", dpi=300)


#%%
df_res_opt_db = pd.DataFrame()
temp_cfg_opt_db = pd.DataFrame()
df_res_opt_theta_alpha = pd.DataFrame()
temp_cfg_opt_theta_alpha = pd.DataFrame()

df_db = pd.DataFrame()
df_theta_alpha = pd.DataFrame()
for kk_zthresh in opt_zthresh:
    for kk_val_duration in opt_val_duration:
        temp_data_db = results_broad.loc[(results_broad["zthresh"]==kk_zthresh) &
                                         (results_broad["val_duration"]==kk_val_duration)]
        median_value=temp_data_db["test_accuracy"].median()

        idx_dummy = temp_data_db["test_accuracy"].idxmin()
        temp_df_db = temp_data_db[temp_data_db.index.isin([idx_dummy])]
        temp_df_db.loc[idx_dummy, 'score'] = median_value

        df_db = pd.concat((df_db, temp_df_db), ignore_index=True)


        temp_data_theta_alpha = results_theta_alpha.loc[(results_theta_alpha["zthresh"]==kk_zthresh) &
                                       (results_theta_alpha["val_duration"]==kk_val_duration)]
        median_value=temp_data_theta_alpha["test_accuracy"].median()

        idx_dummy = temp_data_theta_alpha["test_accuracy"].idxmin()
        temp_df_theta_alpha = temp_data_theta_alpha[temp_data_theta_alpha.index.isin([idx_dummy])]
        temp_df_theta_alpha.loc[idx_dummy, 'test_accuracy'] = median_value

        df_theta_alpha = pd.concat((df_theta_alpha, temp_df_theta_alpha), ignore_index=True)


max_score_db=df_db["test_accuracy"].max()
idx_max_score_db=df_db["test_accuracy"].idxmax()
temp_cfg_opt_db = df_db[df_db.index.isin([idx_max_score_db])]
df_res_opt_db = pd.concat((df_res_opt_db, temp_cfg_opt_db))

max_score_theta_alpha=df_theta_alpha["test_accuracy"].max()
idx_max_score_theta_alpha=df_theta_alpha["test_accuracy"].idxmax()
temp_cfg_opt_theta_alpha = df_theta_alpha[df_theta_alpha.index.isin([idx_max_score_theta_alpha])]
df_res_opt_theta_alpha = pd.concat((df_res_opt_theta_alpha, temp_cfg_opt_theta_alpha))


df_res_opt_db.to_csv(
    path_csv_root + "/SVM/PrelimTests_HC_EP1_MEG_ATM_SVM_ClassificationRebuttal-allnode_rest_BroadBand.csv"
)

df_res_opt_theta_alpha.to_csv(
    path_csv_root + "/SVM/PrelimTests_HC_EP1_MEG_ATM_SVM_ClassificationRebuttal-allnode_rest_theta_alphaBand.csv"
)

#%% best config :
#  - theta-alpha [z=2.6, val_duration:6] - 0.92 median
#  - broad [z=2, val_duration:7] - 0.77 median

# plot
results_theta_alpha_opt=results_theta_alpha[(results_theta_alpha["zthresh"]==2.6) & (results_theta_alpha["val_duration"]==6)]
results_broad_opt=results_broad[(results_broad["zthresh"]==2) & (results_broad["val_duration"]==7)]

results_opt = pd.concat((results_theta_alpha_opt[:50], results_broad_opt))
plt.style.use("dark_background")
g = sns.catplot(y="test_accuracy",
                x='freq',
                hue="freq",
                kind="swarm",
                height=3, aspect=3,
                data=results_opt)

plt.savefig(path_figures_root + "Opt_HC_EP1_ATM_SVM_Classification_allnode-2class-nbSplits"+str(nbSplit)+"_broad_MEG.pdf", dpi=300)


