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
    aout_2=dict()
    nb_nnzeros = list()
    for iaut in range(len(aout)):
        if len(aout[iaut]) > val_duration:
            aout_2[ifi] = ZBIN[aout[iaut]]
            nb_nnzeros.append(np.count_nonzero(aout_2[ifi]))
            mat += transprob(ZBIN[aout[iaut]], nregions)
            ifi += 1
    mat = mat / ifi
    nb_nnzeros_tot=np.sum(nb_nnzeros)/ZBIN.size
    return mat, aout, nb_nnzeros_tot


def threshold_mat(data, thresh=3):
    current_data = data
    binarized_data = np.where(np.abs(current_data) > thresh, 1, 0)
    return (binarized_data)


def find_avalanches(data, thresh=3, val_duration=2):
    binarized_data = threshold_mat(data, thresh=thresh)
    N = binarized_data.shape[0]
    mat, aout, nb_nnzeros_tot = Transprob(binarized_data.T, N, val_duration)
    aout = np.array(aout, dtype=object)
    list_length = [len(i) for i in aout]
    unique_sizes = set(list_length)
    min_size, max_size = min(list_length), max(list_length)
    list_avalanches_bysize = {i: [] for i in unique_sizes}
    for s in aout:
        n = len(s)
        list_avalanches_bysize[n].append(s)
    return (nb_nnzeros_tot, binarized_data, aout, min_size, max_size, list_avalanches_bysize, mat)

#%% parameters for classification
# shuffle order of epochs & labels
rng = np.random.default_rng(42)  # set a random seed
n_perms = 1#00  # number of permutations wanted

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

#%% Compute number of non zeros elements used for classification - classification HC vs EP1 - 32 vs 32
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
nbSplit = 50

lst = []
for f in freqbands:
    fmin = freqbands[f][0]
    fmax = freqbands[f][1]
    freq = str(fmin)+'-'+str(fmax)
    epochs2use=epochs_concat[f].drop([2, 31, 32, 33, 34], reason='USER') # to have the same nb of pz/subjects and because data from the second patient may be corrupted

    data = epochs2use.get_data()
    label_shuffle = labels_concat[f]
    nb_trials = len(data)  # nb subjects/ pz here
    nb_ROIs = np.shape(data)[1]

    # retrieve classification performance
    results_global = pd.read_csv(
            path_csv_root + "/SVM/HC_EP1_IndivOpt_ATM_PLV_Comparison_SVM_Classification-allnode-2class-" +
            "-freq-" + str(fmin) + '-' + str(fmax) + '-nbSplit' + str(nbSplit) + ".csv"
        )
    results_atm_nodal=results_global[results_global["pipeline"]=="ATM+SVM-nodal"]
    results_atm_edges = results_global[results_global["pipeline"] == "ATM+SVM"]

    ### ATM + SVM -- attempt to optimize the code to make it faster...
    temp = np.transpose(data, (1, 0, 2))
    temp_nc = np.reshape(temp, (np.shape(temp)[0], np.shape(temp)[1] * np.shape(temp)[2]))
    zscored_data = zscore(temp_nc, axis=1)
    # epoching here before computing the avalanches
    temp_zscored_data_ep = np.reshape(zscored_data,
                                      (np.shape(temp)[0], np.shape(temp)[1], np.shape(temp)[2]))
    zscored_data_ep = np.transpose(temp_zscored_data_ep, (1, 0, 2))


    for kk_zthresh in opt_zthresh:
        for kk_val_duration in opt_val_duration:
            binarized_recording = np.empty((nb_trials, nb_ROIs, 100865))
            nb_nnzeros_tot_recording = np.empty(nb_trials)
            temp_results_edges = results_atm_edges[(results_atm_edges["zthresh"]==kk_zthresh) & (results_atm_edges["val_duration"]==kk_val_duration)]
            temp_results_nodal = results_atm_nodal[
                (results_atm_nodal["zthresh"] == kk_zthresh) & (results_atm_nodal["val_duration"] == kk_val_duration)]

            for kk_trial in range(nb_trials):
                nb_nnzeros_tot, binarized_data, list_avalanches, min_size_avalanches, max_size_avalanches, list_avalanches_bysize, temp_ATM = find_avalanches(
                    zscored_data_ep[kk_trial, :, :], thresh=kk_zthresh, val_duration=kk_val_duration)
                # ATM: nb_trials x nb_ROIs x nb_ROIs matrix
                binarized_recording[kk_trial, :, :] = binarized_data
                nb_nnzeros_tot_recording[kk_trial]=nb_nnzeros_tot

            accuracy_edges = temp_results_edges["test_accuracy"].median()
            accuracy_nodal = temp_results_nodal["test_accuracy"].median()
            ratio_edges=np.mean(nb_nnzeros_tot_recording)/accuracy_edges
            ratio_nodal = np.mean(nb_nnzeros_tot_recording) / accuracy_nodal
            lst.append([kk_zthresh, kk_val_duration, freq, np.mean(nb_nnzeros_tot_recording), accuracy_edges, accuracy_nodal, ratio_edges, ratio_nodal])


cols = ["zthresh","aval_duration","freq", "nb_bits_ratio", "median_accuracy_edge", "median_accuracy_node", "ratio_edges", "ratio_nodal"]
pd_nb_bits_ATM = pd.DataFrame(lst, columns=cols)

pd_nb_bits_ATM.to_csv(
            path_csv_root + "/SVM/HC_EP1_BitEffect_ATM_Classification-allnode-2class-" +
            ".csv"
        )
print(
            "saved " +
            path_csv_root + "/SVM/HC_EP1_BitEffect_ATM_Classification-allnode-2class-" +
             ".csv"
        )

#%% prova di plot
import matplotlib.pyplot as plt
import seaborn as sns
path_figures_root = "/Users/marieconstance.corsi/Documents/GitHub/Fenicotteri-equilibristi/Figures/Classification/"


pd_nb_bits_ATM = pd.read_csv(
            path_csv_root + "/SVM/HC_EP1_BitEffect_ATM_Classification-allnode-2class-" +
            ".csv"
        )

list_freq=pd_nb_bits_ATM["freq"].unique()
for frequency in list_freq:
    temp_pd_nb_bits_ATM = pd_nb_bits_ATM[pd_nb_bits_ATM["freq"]==frequency]
    sns.catplot(data = temp_pd_nb_bits_ATM, x="zthresh", y="ratio_edges",
                hue="aval_duration",# col_wrap=2,
                kind='swarm', s= 4, height=4,aspect=3)
    plt.savefig(path_figures_root + "OctetEffect_HC_EP1_IndivOpt_PLV_SVM_Classification_edges_2class_"+frequency+"_MEG.pdf", dpi=300)

    sns.catplot(data = temp_pd_nb_bits_ATM, x="zthresh", y="ratio_nodal",
                hue="aval_duration",# col_wrap=2,
                kind='swarm', s= 4, height=4,aspect=3)
    plt.savefig(path_figures_root + "OctetEffect_HC_EP1_IndivOpt_PLV_SVM_Classification_nodal_2class_"+frequency+"_MEG.pdf", dpi=300)
