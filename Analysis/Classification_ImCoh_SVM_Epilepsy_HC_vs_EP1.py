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

freqbands = {'theta': [3, 8],
             'alpha': [8, 14],
             'theta-alpha': [3, 14],  # Epilepsy case
             'paper': [3, 40]}

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

results_atm = pd.DataFrame()
results_ImCoh = pd.DataFrame()

for f in freqbands:
    fmin = freqbands[f][0]
    fmax = freqbands[f][1]

    epochs2use=epochs_concat[f].drop([2, 31, 32, 33, 34], reason='USER') # to have the same nb of pz/subjects
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
        mat_ImCoh = preproc_meg.fit_transform(epochs_in_permuted_order) # actually unfair because does an average of the estimations ie more robust than ATM this way

        mat_true = np.ones((68, 68), dtype=bool)
        tril_mat_true = np.tril(mat_true, k=-1)
        mat_tril_ImCoh = np.empty((nb_trials, 2278))
        for kk_trial in range(nb_trials):
            temp=mat_ImCoh[kk_trial,:,:]
            mat_tril_ImCoh[kk_trial,:] = temp[tril_mat_true]
        ImCoh_cl = mat_tril_ImCoh # already flatten

        #ImCoh_cl = np.reshape(mat_tril_ImCoh, (nb_trials, nb_ROIs * nb_ROIs))
       # ImCoh_nodal_cl = np.sum(ImCoh_cl,1)

        clf_2 = Pipeline([('SVM', svm)])
        # score_ImCoh_SVM = cross_val_score(clf_2, ImCoh_cl, labels, cv=cv, n_jobs=None)
        scores_ImCoh_SVM = cross_validate(clf_2, ImCoh_cl,  label_shuffle, cv=cv, n_jobs=None, scoring=scoring, return_estimator=False) # returns the estimator objects for each cv split!
        #scores_ImCoh_nodal_SVM = cross_validate(clf_2, ImCoh_nodal_cl,  label_shuffle, cv=cv, n_jobs=None, scoring=scoring, return_estimator=False) # returns the estimator objects for each cv split!

        # concatenate ImCoh results in a dedicated dataframe
        pd_ImCoh_SVM = pd.DataFrame.from_dict(scores_ImCoh_SVM)
        temp_results_ImCoh = pd_ImCoh_SVM
        ppl_ImCoh = ["ImCoh+SVM"] * len(pd_ImCoh_SVM)
        temp_results_ImCoh["pipeline"] = ppl_ImCoh
        temp_results_ImCoh["split"] = [nbSplit] * len(ppl_ImCoh)
        temp_results_ImCoh["freq"] = [str(fmin) + '-' + str(fmax)] * len(ppl_ImCoh)

        # pd_ImCoh_nodal_SVM = pd.DataFrame.from_dict(scores_ImCoh_nodal_SVM)
        # temp_results_ImCoh_nodal = pd_ImCoh_nodal_SVM
        # ppl_ImCoh_nodal = ["ImCoh+SVM-nodal"] * len(pd_ImCoh_nodal_SVM)
        # temp_results_ImCoh_nodal["pipeline"] = ppl_ImCoh_nodal
        # temp_results_ImCoh_nodal["split"] = [nbSplit] * len(ppl_ImCoh_nodal)
        # temp_results_ImCoh_nodal["freq"] = [str(fmin) + '-' + str(fmax)] * len(ppl_ImCoh_nodal)

        results_ImCoh = pd.concat((results_ImCoh, temp_results_ImCoh))#, temp_results_ImCoh_nodal))

    # concatenate results in a single dataframe
    results_global = results_ImCoh
    results_global.to_csv(
            path_csv_root + "/SVM/Tril_HC_EP1_IndivOpt_ImCoh_SVM_Classification-allnode-2class-" +
            "-freq-" + str(fmin) + '-' + str(fmax) + '-nbSplit' + str(nbSplit) + ".csv"
        )
    print(
            "saved " +
            path_csv_root + "/SVM/Tril_HC_EP1_IndivOpt_ImCoh_SVM_Classification-allnode-2class-" +
            "-freq-" + str(fmin) + '-' + str(fmax) + '-nbSplit' + str(nbSplit) + ".csv"
        )


#%%
results_ImCoh_alone=pd.DataFrame()

for f in freqbands:
    fmin = freqbands[f][0]
    fmax = freqbands[f][1]
    temp_results = pd.read_csv(         path_csv_root + "/SVM/Tril_HC_EP1_IndivOpt_ImCoh_SVM_Classification-allnode-2class-" +
            "-freq-" + str(fmin) + '-' + str(fmax) + '-nbSplit' + str(nbSplit) + ".csv"
             )

    results_ImCoh_alone = pd.concat((results_ImCoh_alone, temp_results))


#%%
import matplotlib.pyplot as plt
import seaborn as sns
path_figures_root = "/Users/marieconstance.corsi/Documents/GitHub/Fenicotteri-equilibristi/Figures/Classification/"
# pb in the loop directly use the alpha case that already contains all
results_ImCoh_alone2use=temp_results
plt.style.use("classic")
g = sns.catplot(y="test_accuracy",
                x='freq',
                hue="freq",
                kind="swarm",
                #row="freq",
                height=7, aspect=2,
                data=results_ImCoh_alone2use)
plt.savefig(path_figures_root + "Tril_HC_EP1_IndivOpt_ImCoh_SVM_Classification_edges_nodes-2class-nbSplits"+str(nbSplit)+"_broad_MEG.pdf", dpi=300)
