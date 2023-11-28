# %%
"""
==========================================================================================================================================================================================
Attempt to classify EEG data in the source space from temporal lobe epilepsy patients vs healthy subjects - neuronal avalanches vs imaginary coherence +SVM
==========================================================================================================================================================================================

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

from scipy.stats import zscore

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_validate

# to compute ImCoh estimations for each trial
from Analysis.fc_pipeline import FunctionalTransformer
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

#%% parameters for classification
nbSplit = 50
scoring = ['precision', 'recall', 'accuracy', 'f1', 'roc_auc']

# shuffle order of epochs & labels
rng = np.random.default_rng(42)  # set a random seed
n_perms = 1  # number of permutations wanted

# parameters for the default classifier & cross-validation
svm = GridSearchCV(SVC(), {"kernel": ("linear", "rbf"), "C": [0.1, 1, 10]}, cv=5)
cv = ShuffleSplit(nbSplit, test_size=0.2, random_state=21)


# %% parameters to be applied to extract the features
freqbands = {'theta-alpha': [3, 14],  
             'broad': [3, 40]}

opt_zthresh = [1.4, 1.6, 1.8, 2, 2.2, 2.4, 2.6, 2.8, 3]
opt_val_duration = [2, 3, 4, 5, 6, 7, 8]

opt_trial_duration = [100864/256] # one single trial for everything
fs = 256

test=mat73.loadmat('/Users/marieconstance.corsi/Documents/GitHub/NeuronalAvalanches_TemporalLobeEpilepsy_EEG/Database/1_Clinical/Epilepsy_GMD/Data_Epi_MEG_4Classif_concat_NoTrials.mat')
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

for f in freqbands:
    fmin = freqbands[f][0]
    fmax = freqbands[f][1]
    results_global = pd.DataFrame()
    results_atm = pd.DataFrame()
    results_ImCoh = pd.DataFrame()
    epochs2use=epochs_concat[f].drop([2, 31, 32, 33, 34], reason='USER') # to have the same nb of pz/subjects and because data from the second patient may be corrupted
    for iperm in range(n_perms): # in case we want to do some permutations to increase the statistical power
        perm = rng.permutation(len(epochs2use))
        epochs_in_permuted_order = epochs2use[perm]
        data_shuffle = epochs_in_permuted_order.get_data()
        label_shuffle = [labels_concat[f][x] for x in perm]
        nb_trials = len(data_shuffle)  # nb subjects/ pz here
        nb_ROIs = np.shape(data_shuffle)[1]

        ### ImCoh + SVM
        # # here to compute FC on long recordings we increase the time window of interest to speed up the process
        ft = FunctionalTransformer(
            delta=30, ratio=0.5, method="imcoh", fmin=fmin, fmax=fmax
        )
        preproc_meg = Pipeline(steps=[("ft", ft)])  # , ("spd", EnsureSPD())])
        mat_ImCoh = preproc_meg.fit_transform(epochs_in_permuted_order)

        ImCoh_cl = np.reshape(mat_ImCoh, (nb_trials, nb_ROIs * nb_ROIs))
        ImCoh_nodal_cl = np.sum(mat_ImCoh,1)

        clf_2 = Pipeline([('SVM', svm)])
        scores_ImCoh_SVM = cross_validate(clf_2, ImCoh_cl,  label_shuffle, cv=cv, n_jobs=None, scoring=scoring, return_estimator=False) # returns the estimator objects for each cv split!
        scores_ImCoh_nodal_SVM = cross_validate(clf_2, ImCoh_nodal_cl,  label_shuffle, cv=cv, n_jobs=None, scoring=scoring, return_estimator=False) # returns the estimator objects for each cv split!

        # concatenate ImCoh results in a dedicated dataframe
        pd_ImCoh_SVM = pd.DataFrame.from_dict(scores_ImCoh_SVM)
        temp_results_ImCoh = pd_ImCoh_SVM
        ppl_ImCoh = ["ImCoh+SVM"] * len(pd_ImCoh_SVM)
        temp_results_ImCoh["pipeline"] = ppl_ImCoh
        temp_results_ImCoh["split"] = [nbSplit] * len(ppl_ImCoh)
        temp_results_ImCoh["freq"] = [str(fmin) + '-' + str(fmax)] * len(ppl_ImCoh)

        pd_ImCoh_nodal_SVM = pd.DataFrame.from_dict(scores_ImCoh_nodal_SVM)
        temp_results_ImCoh_nodal = pd_ImCoh_nodal_SVM
        ppl_ImCoh_nodal = ["ImCoh+SVM-nodal"] * len(pd_ImCoh_nodal_SVM)
        temp_results_ImCoh_nodal["pipeline"] = ppl_ImCoh_nodal
        temp_results_ImCoh_nodal["split"] = [nbSplit] * len(ppl_ImCoh_nodal)
        temp_results_ImCoh_nodal["freq"] = [str(fmin) + '-' + str(fmax)] * len(ppl_ImCoh_nodal)

        results_ImCoh = pd.concat((results_ImCoh, temp_results_ImCoh, temp_results_ImCoh_nodal))


        ### ATM + SVM
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