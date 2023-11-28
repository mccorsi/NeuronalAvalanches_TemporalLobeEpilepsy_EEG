# %%
"""
==============================================================
Attempt to classify MEG data in the source space - neuronal avalanches vs classical approaches - classification on longer trials w/ SVM - lower scales - broadband
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

import random
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

def _build_df_plv_from_scores(scores, ppl_name, nbSplit, fmin, fmax, duration, permutation):
    pd_PLV_classif_res = pd.DataFrame.from_dict(scores)
    ppl_plv = [ppl_name] * len(pd_PLV_classif_res)
    pd_PLV_classif_res["pipeline"] = ppl_plv
    pd_PLV_classif_res["split"] = [nbSplit] * len(ppl_plv)
    pd_PLV_classif_res["freq"] = [str(fmin) + '-' + str(fmax)] * len(ppl_plv)
    pd_PLV_classif_res["trial-duration"] = [duration] * len(ppl_plv)
    pd_PLV_classif_res["trial-permutation"] = [permutation] * len(ppl_plv)
    pd_PLV_classif_res["split"] = range(0, nbSplit)

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
# freqbands = {'theta': [3, 8],
#              'alpha': [8, 14],
#              'theta-alpha': [3, 14],  # Epilepsy case
#              'paper': [3, 40]}

freqbands = {'paper': [3, 40]}

#opt_trial_duration = [5, 10, 15, 30, 60, 120, 180, 300]#, 100864/256] # one single trial for everything - in seconds
opt_trial_duration = [0.050, 0.100, 0.200, 0.500, 1, 2]
fs = 256

test=mat73.loadmat('/Users/marieconstance.corsi/Documents/GitHub/Fenicotteri-equilibristi/Database/1_Clinical/Epilepsy_GMD/Data_Epi_MEG_4Classif_concat_NoTrials.mat')
ch_names = test['labels_AAL1']
ch_types = ["eeg" for i in range(np.shape(ch_names)[0])]

#%% retrieve optimal parameters, for each freq band [zthresh, aval_dur]
opt_atm_param_edge = dict()
opt_atm_param_nodal = dict()
perf_opt_atm_param_edge = dict()
perf_opt_atm_param_nodal = dict()

df_res_opt_db=pd.read_csv(
    path_csv_root + "/SVM/OptConfig_HC_EP1_MEG_ATM_SVM_ClassificationRebuttal-allnode_rest_BroadBand.csv"
)
opt_atm_param_edge["paper"] = [df_res_opt_db["zthresh"][0], df_res_opt_db["val_duration"][0]]
perf_opt_atm_param_edge["paper"] = df_res_opt_db["test_accuracy"][0]

df_res_opt_db_nodal=pd.read_csv(
    path_csv_root + "/SVM/OptConfig_HC_EP1_MEG_ATM_SVM_ClassificationRebuttal-nodal_rest_BroadBand.csv"
)
opt_atm_param_nodal["paper"] = [df_res_opt_db_nodal["zthresh"][0], df_res_opt_db_nodal["val_duration"][0]]
perf_opt_atm_param_nodal["paper"] = df_res_opt_db_nodal["test_accuracy"][0]

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
#results_plv = pd.DataFrame()
num_perm = 100
max_time = 100864/256 # recording length
for f in freqbands:
    fmin = freqbands[f][0]
    fmax = freqbands[f][1]
    epochs2use = epochs_concat[f].drop([2, 31, 32, 33, 34],
                                       reason='USER')  # to have the same nb of pz/subjects and because data from the second patient may be corrupted
    for kk_trial_crop in opt_trial_duration:
        for kk_perm in range(num_perm):
            if kk_trial_crop == 0.5: # need of an integer
                start = random.randrange(0, max_time - kk_trial_crop*2)
            elif kk_trial_crop ==0.05:
                start = random.randrange(0, max_time - kk_trial_crop * 20)
            elif kk_trial_crop ==0.10:
                start = random.randrange(0, max_time - kk_trial_crop * 10)
            elif kk_trial_crop == 0.20:
                start = random.randrange(0, max_time - kk_trial_crop * 5)
            else:
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

                # ### PLV + SVM
                # # # here to compute FC on long recordings we increase the time window of interest to speed up the process
                # ft = FunctionalTransformer(
                #     delta=10, ratio=0.5, method="plv", fmin=fmin, fmax=fmax
                # )
                # preproc_meg = Pipeline(steps=[("ft", ft)])  # , ("spd", EnsureSPD())])
                # mat_PLV = preproc_meg.fit_transform(epochs_in_permuted_order) # actually unfair because does an average of the estimations ie more robust than ATM this way
                #
                # PLV_cl = np.reshape(mat_PLV, (nb_trials, nb_ROIs * nb_ROIs))
                # PLV_nodal_cl = np.sum(mat_PLV,1)
                #
                # clf_2 = Pipeline([('SVM', svm)])
                # # score_PLV_SVM = cross_val_score(clf_2, PLV_cl, labels, cv=cv, n_jobs=None)
                # scores_PLV_SVM = cross_validate(clf_2, PLV_cl,  label_shuffle, cv=cv, n_jobs=None, scoring=scoring, return_estimator=False) # returns the estimator objects for each cv split!
                # scores_PLV_nodal_SVM = cross_validate(clf_2, PLV_nodal_cl,  label_shuffle, cv=cv, n_jobs=None, scoring=scoring, return_estimator=False) # returns the estimator objects for each cv split!
                #
                # # concatenate PLV results in a dedicated dataframe
                # temp_results_plv_nodal = _build_df_plv_from_scores(scores=scores_PLV_nodal_SVM, ppl_name="PLV+SVM-nodal",
                #                                                    nbSplit=nbSplit, fmin=fmin, fmax=fmax, duration=kk_trial_crop)
                # temp_results_plv = _build_df_plv_from_scores(scores=scores_PLV_SVM, ppl_name="PLV+SVM",
                #                                              nbSplit=nbSplit, fmin=fmin, fmax=fmax, duration=kk_trial_crop)
                # results_plv = pd.concat((results_plv, temp_results_plv, temp_results_plv_nodal), ignore_index=True)

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
results_global = results_atm #pd.concat((results_atm, results_plv))
results_global.to_csv(
        path_csv_root + "/SVM/TrialDurationEffect_HC_EP1_IndivOpt_ATM__SVM_Classification-allnode-2class-" +
                '-nbSplit' + str(nbSplit) + "_LowerScale.csv"
        )
print(
        "saved " +
        path_csv_root + "/SVM/TrialDurationEffect_HC_EP1_IndivOpt_ATM__SVM_Classification-allnode-2class-" +
        '-nbSplit' + str(nbSplit) + "_LowerScale.csv"
        )


#%% plots results
import matplotlib.pyplot as plt
import seaborn as sns
path_figures_root = "/Users/marieconstance.corsi/Documents/GitHub/Fenicotteri-equilibristi/Figures/Classification/"


results_trial_dur_HigherScale= pd.read_csv(path_csv_root + "/SVM/TrialDurationEffect_HC_EP1_IndivOpt_ATM__SVM_Classification-allnode-2class-" +
        '-nbSplit' + str(nbSplit) + ".csv"
             )
results_trial_dur_LowerScale = pd.read_csv(path_csv_root + "/SVM/TrialDurationEffect_HC_EP1_IndivOpt_ATM__SVM_Classification-allnode-2class-" +
                '-nbSplit' + str(nbSplit) + "_LowerScale.csv"
)

results_trial_dur = pd.concat((results_trial_dur_HigherScale,results_trial_dur_LowerScale))

results_atm_edges = results_trial_dur[results_trial_dur["pipeline"] == "ATM+SVM"]
results_atm_nodal = results_trial_dur[results_trial_dur["pipeline"] == "ATM+SVM-nodal"]

opt_trial_duration=results_trial_dur['trial-duration'].unique()
# study by split
lst = list()
lst_nodal = list()
for f in freqbands:
    fmin = freqbands[f][0]
    fmax = freqbands[f][1]
    results_atm_edges_f= results_atm_edges[results_atm_edges["freq"] == str(fmin)+'-'+str(fmax)]
    results_atm_nodal_f = results_atm_nodal[results_atm_nodal["freq"] == str(fmin) + '-' + str(fmax)]
    # for a given duration: for each split, median over all the possible combinations of trial
    for kk_trial_crop in opt_trial_duration:
        temp_results_atm_edges_trial_dur = results_atm_edges_f[results_atm_edges_f["trial-duration"]==kk_trial_crop]
        temp_results_atm_nodal_trial_dur = results_atm_nodal_f[results_atm_nodal_f["trial-duration"] == kk_trial_crop]

        for kk_split in range(nbSplit):
            temp_res_atm_edges = temp_results_atm_edges_trial_dur[temp_results_atm_edges_trial_dur["split"]==kk_split]
            score_edges = temp_res_atm_edges["test_accuracy"].median()
            pipeline_edges = temp_res_atm_edges["pipeline"].unique()[0]
            zthresh_edges  = temp_res_atm_edges["zthresh"].unique()[0]
            val_duration_edges  = temp_res_atm_edges["val_duration"].unique()[0]
            trial_duration_edges = temp_res_atm_edges["trial-duration"].unique()[0]
            freq_edges = temp_res_atm_edges["freq"].unique()[0]
            lst.append([score_edges, pipeline_edges, kk_split, zthresh_edges, val_duration_edges, freq_edges, trial_duration_edges])

            temp_res_atm_nodal = temp_results_atm_nodal_trial_dur[temp_results_atm_nodal_trial_dur["split"]==kk_split]
            score_nodal = temp_res_atm_nodal["test_accuracy"].median()
            pipeline_nodal = temp_res_atm_nodal["pipeline"].unique()[0]
            zthresh_nodal  = temp_res_atm_nodal["zthresh"].unique()[0]
            val_duration_nodal  = temp_res_atm_nodal["val_duration"].unique()[0]
            trial_duration_nodal = temp_res_atm_nodal["trial-duration"].unique()[0]
            freq_nodal = temp_res_atm_nodal["freq"].unique()[0]
            lst_nodal.append([score_nodal, pipeline_nodal, kk_split, zthresh_nodal, val_duration_nodal, freq_nodal, trial_duration_nodal])


cols = ["test_accuracy", "pipeline", "split", "zthresh","aval_duration", "freq", "trial_duration"]
pd_nb_bits_ATM_bySplit_edges = pd.DataFrame(lst, columns=cols)
pd_nb_bits_ATM_bySplit_nodal = pd.DataFrame(lst_nodal, columns=cols)


# study by perm for a given definition of the trial
lst = list()
lst_nodal = list()
num_perm=100
for f in freqbands:
    fmin = freqbands[f][0]
    fmax = freqbands[f][1]
    results_atm_edges_f= results_atm_edges[results_atm_edges["freq"] == str(fmin)+'-'+str(fmax)]
    results_atm_nodal_f = results_atm_nodal[results_atm_nodal["freq"] == str(fmin) + '-' + str(fmax)]

    # for a given duration: for each split, median over all the possible combinations of trial
    for kk_trial_crop in opt_trial_duration:
        temp_results_atm_edges_trial_dur = results_atm_edges_f[results_atm_edges_f["trial-duration"]==kk_trial_crop]
        temp_results_atm_nodal_trial_dur = results_atm_nodal_f[results_atm_nodal_f["trial-duration"] == kk_trial_crop]

        for kk_trial_perm in range(num_perm):
            temp_res_atm_edges = temp_results_atm_edges_trial_dur[temp_results_atm_edges_trial_dur["trial-permutation"]==kk_trial_perm]
            score_edges = temp_res_atm_edges["test_accuracy"].median()
            pipeline_edges = temp_res_atm_edges["pipeline"].unique()[0]
            zthresh_edges  = temp_res_atm_edges["zthresh"].unique()[0]
            val_duration_edges  = temp_res_atm_edges["val_duration"].unique()[0]
            trial_duration_edges = temp_res_atm_edges["trial-duration"].unique()[0]
            freq_edges = temp_res_atm_edges["freq"].unique()[0]
            lst.append([score_edges, pipeline_edges, kk_trial_perm, zthresh_edges, val_duration_edges, freq_edges, trial_duration_edges])

            temp_res_atm_nodal = temp_results_atm_nodal_trial_dur[temp_results_atm_nodal_trial_dur["trial-permutation"]==kk_trial_perm]
            score_nodal = temp_res_atm_nodal["test_accuracy"].median()
            pipeline_nodal = temp_res_atm_nodal["pipeline"].unique()[0]
            zthresh_nodal  = temp_res_atm_nodal["zthresh"].unique()[0]
            val_duration_nodal  = temp_res_atm_nodal["val_duration"].unique()[0]
            trial_duration_nodal = temp_res_atm_nodal["trial-duration"].unique()[0]
            freq_nodal = temp_res_atm_nodal["freq"].unique()[0]
            lst_nodal.append([score_nodal, pipeline_nodal, kk_trial_perm, zthresh_nodal, val_duration_nodal, freq_nodal, trial_duration_nodal])

cols = ["test_accuracy", "pipeline", "trial-permutation", "zthresh","aval_duration", "freq", "trial_duration"]
pd_nb_bits_ATM_byTrialPerm_edges = pd.DataFrame(lst, columns=cols)
pd_nb_bits_ATM_byTrialPerm_nodal = pd.DataFrame(lst_nodal, columns=cols)

#%% plot results
list_freq=pd_nb_bits_ATM_bySplit_edges["freq"].unique()
plt.style.use("classic")
for frequency in list_freq:
    temp_pd_nb_bits_ATM_bySplit_edges = pd_nb_bits_ATM_bySplit_edges[pd_nb_bits_ATM_bySplit_edges["freq"]==frequency]
    g = sns.catplot(y="test_accuracy",
                    x='trial_duration',
                    #row='freq',
                    #row='pipeline',
                    #hue="pipeline",
                    kind="swarm",
                    #height=9, aspect=4, s=6,
                    height=9, s=6, aspect=9,
                    data=temp_pd_nb_bits_ATM_bySplit_edges)
    plt.savefig(path_figures_root + "TrialDurationEffect_BySplit_HC_EP1_ATM_SVM_Classification_edges-2class-nbSplits"+str(nbSplit)+"_"+frequency+"_MEG.pdf", dpi=300)

    temp_pd_nb_bits_ATM_bySplit_nodal = pd_nb_bits_ATM_bySplit_nodal[pd_nb_bits_ATM_bySplit_nodal["freq"] == frequency]
    g = sns.catplot(y="test_accuracy",
                    x='trial_duration',
                    #row='freq',
                    #row='pipeline',
                    #hue="pipeline",
                    kind="swarm",
                    #height=9, aspect=4, s=6,
                    height=9, s=6, aspect=9,
                    data=pd_nb_bits_ATM_bySplit_nodal)
    plt.savefig(path_figures_root + "TrialDurationEffect_BySplit_HC_EP1_ATM_SVM_Classification_nodal-2class-nbSplits"+str(nbSplit)+"_"+frequency+"_MEG.pdf", dpi=300)


    temp_pd_nb_bits_ATM_byTrialPerm_edges = pd_nb_bits_ATM_byTrialPerm_edges[pd_nb_bits_ATM_byTrialPerm_edges["freq"]==frequency]
    g = sns.catplot(y="test_accuracy",
                    x='trial_duration',
                    #row='freq',
                    #row='pipeline',
                    #hue="pipeline",
                    kind="swarm",
                    #height=9, aspect=4, s=6,
                    height=9, s=6, aspect=10,
                    data=pd_nb_bits_ATM_byTrialPerm_edges)
    plt.savefig(path_figures_root + "TrialDurationEffect_ByTrialPerm_HC_EP1_ATM_SVM_Classification_edges-2class-nbSplits"+str(nbSplit)+"_"+frequency+"_MEG.pdf", dpi=300)

    temp_pd_nb_bits_ATM_byTrialPerm_nodal = pd_nb_bits_ATM_byTrialPerm_nodal[pd_nb_bits_ATM_byTrialPerm_nodal["freq"]==frequency]
    g = sns.catplot(y="test_accuracy",
                    x='trial_duration',
                    #row='freq',
                    #row='pipeline',
                    #hue="pipeline",
                    kind="swarm",
                    #height=9, aspect=4, s=6,
                    height=10, s=6, aspect=11,
                    data=temp_pd_nb_bits_ATM_byTrialPerm_nodal)
    plt.savefig(path_figures_root + "TrialDurationEffect_ByTrialPerm_HC_EP1_ATM_SVM_Classification_nodal-2class-nbSplits"+str(nbSplit)+"_"+frequency+"_MEG.pdf", dpi=300)