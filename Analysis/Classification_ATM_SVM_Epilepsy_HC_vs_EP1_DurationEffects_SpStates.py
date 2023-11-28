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
from mne_connectivity.viz.circle import _plot_connectivity_circle
from mne.viz import circular_layout#, plot_connectivity_circle

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


import math
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


def _build_df_atm_from_scores(scores, ppl_name, nbSplit, zthresh_value, aval_duration_value, fmin, fmax, duration):
    pd_ATM_classif_res = pd.DataFrame.from_dict(scores)
    ppl_atm = [ppl_name] * len(pd_ATM_classif_res)
    pd_ATM_classif_res["pipeline"] = ppl_atm
    pd_ATM_classif_res["split"] = [nbSplit] * len(ppl_atm)
    pd_ATM_classif_res["zthresh"] = [zthresh_value] * len(ppl_atm)
    pd_ATM_classif_res["val_duration"] = [aval_duration_value] * len(ppl_atm)
    pd_ATM_classif_res["freq"] = [str(fmin) + '-' + str(fmax)] * len(ppl_atm)
    pd_ATM_classif_res["trial-duration"] = [duration] * len(ppl_atm)

    return  pd_ATM_classif_res

def _build_df_plv_from_scores(scores, ppl_name, nbSplit, fmin, fmax, duration):
    pd_PLV_classif_res = pd.DataFrame.from_dict(scores)
    ppl_plv = [ppl_name] * len(pd_PLV_classif_res)
    pd_PLV_classif_res["pipeline"] = ppl_plv
    pd_PLV_classif_res["split"] = [nbSplit] * len(ppl_plv)
    pd_PLV_classif_res["freq"] = [str(fmin) + '-' + str(fmax)] * len(ppl_plv)
    pd_PLV_classif_res["trial-duration"] = [duration] * len(ppl_plv)

    return  pd_PLV_classif_res

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

def isSquare (m): return all (len (row) == len (m) for row in m)

def regions_bst_to_lobes( argument ):
    '''
        gather brainstorm labels in 5 lobes/areas
    '''
    switcher = {
        # gathers Left and Right in one single lobe
        # Frontal: includes both Frontal and PreFrontal in this way
        'LF': 'F' ,
        'RF': 'F' ,
        'LPF': 'F' ,
        'RPF': 'F' ,
        # Central: C --> Motor: M, except postcentral that is going to be placed in parietal...
        'LC': 'M' ,
        'RC': 'M' ,
        'M': 'M', # in case that already done
        # Temporal: includes both Temporal and Lateral
        'LT': 'T' ,
        'RT': 'T' ,
        'LL': 'T' ,
        'RL': 'T' ,
        # Parietal
        'LP': 'P' ,
        'RP': 'P' ,
        'P': 'P',  # in case that already done
        # Occipital
        'LO': 'O' ,
        'RO': 'O'
    }
    return switcher.get ( argument )

def Lobes_Partition( MatrixToBeDivided , idx_F , idx_M , idx_P , idx_T , idx_O ):
    SubMatrix_F_F = MatrixToBeDivided[ np.ix_ ( idx_F , idx_F ) ]
    SubMatrix_F_M = MatrixToBeDivided[ np.ix_ ( idx_F , idx_M ) ]
    SubMatrix_F_T = MatrixToBeDivided[ np.ix_ ( idx_F , idx_T ) ]
    SubMatrix_F_P = MatrixToBeDivided[ np.ix_ ( idx_F , idx_P ) ]
    SubMatrix_F_O = MatrixToBeDivided[ np.ix_ ( idx_F , idx_O ) ]

    # SubMatrix_M_F=MatrixToBeDivided[np.ix_(idx_M,idx_F)]
    SubMatrix_M_M = MatrixToBeDivided[ np.ix_ ( idx_M , idx_M ) ]
    SubMatrix_M_T = MatrixToBeDivided[ np.ix_ ( idx_M , idx_T ) ]
    SubMatrix_M_P = MatrixToBeDivided[ np.ix_ ( idx_M , idx_P ) ]
    SubMatrix_M_O = MatrixToBeDivided[ np.ix_ ( idx_M , idx_O ) ]

    # SubMatrix_T_F=MatrixToBeDivided[np.ix_(idx_T,idx_F)]
    # SubMatrix_T_M=MatrixToBeDivided[np.ix_(idx_T,idx_M)]
    SubMatrix_T_T = MatrixToBeDivided[ np.ix_ ( idx_T , idx_T ) ]
    SubMatrix_T_P = MatrixToBeDivided[ np.ix_ ( idx_T , idx_P ) ]
    SubMatrix_T_O = MatrixToBeDivided[ np.ix_ ( idx_T , idx_O ) ]

    # SubMatrix_P_F=MatrixToBeDivided[np.ix_(idx_P,idx_F)]
    # SubMatrix_P_M=MatrixToBeDivided[np.ix_(idx_P,idx_M)]
    # SubMatrix_P_T=MatrixToBeDivided[np.ix_(idx_P,idx_T)]
    SubMatrix_P_P = MatrixToBeDivided[ np.ix_ ( idx_P , idx_P ) ]
    SubMatrix_P_O = MatrixToBeDivided[ np.ix_ ( idx_P , idx_O ) ]

    # SubMatrix_O_F=MatrixToBeDivided[np.ix_(idx_O,idx_F)]
    # SubMatrix_O_M=MatrixToBeDivided[np.ix_(idx_O,idx_M)]
    # SubMatrix_O_T=MatrixToBeDivided[np.ix_(idx_O,idx_T)]
    # SubMatrix_O_P=MatrixToBeDivided[np.ix_(idx_O,idx_P)]
    SubMatrix_O_O = MatrixToBeDivided[ np.ix_ ( idx_O , idx_O ) ]

    temp1 = np.hstack((SubMatrix_F_F, SubMatrix_F_M, SubMatrix_F_T, SubMatrix_F_P, SubMatrix_F_O))
    temp2 = np.hstack((np.transpose(SubMatrix_F_M), SubMatrix_M_M, SubMatrix_M_T, SubMatrix_M_P, SubMatrix_M_O))
    temp1b = np.vstack((temp1, temp2))
    temp3 = np.hstack(
        (np.transpose(SubMatrix_F_T), np.transpose(SubMatrix_M_T), SubMatrix_T_T, SubMatrix_T_P, SubMatrix_T_O))
    temp4 = np.hstack((np.transpose(SubMatrix_F_P), np.transpose(SubMatrix_M_P), np.transpose(SubMatrix_T_P),
                       SubMatrix_P_P, SubMatrix_P_O))
    temp3b = np.vstack((temp3, temp4))
    temp4b = np.vstack((temp1b, temp3b))
    temp5 = np.hstack((np.transpose(SubMatrix_F_O), np.transpose(SubMatrix_M_O), np.transpose(SubMatrix_T_O),
                       np.transpose(SubMatrix_P_O), SubMatrix_O_O))
    output = np.vstack((temp4b, temp5))

    return SubMatrix_F_F , SubMatrix_F_M , SubMatrix_F_T , SubMatrix_F_P , SubMatrix_F_O , \
           SubMatrix_M_M , SubMatrix_M_T , SubMatrix_M_P , SubMatrix_M_O , \
           SubMatrix_T_T , SubMatrix_T_P , SubMatrix_T_O , \
           SubMatrix_P_P , SubMatrix_P_O , \
           SubMatrix_O_O, output

#%% lobes - DK
ROI_DK_list = [ 'bankssts L' , 'bankssts R' , 'caudalanteriorcingulate L' , 'caudalanteriorcingulate R' ,
                'caudalmiddlefrontal L' , 'caudalmiddlefrontal R' , 'cuneus L' , 'cuneus R' , 'entorhinal L' ,
                'entorhinal R' , 'frontalpole L' , 'frontalpole R' , 'fusiform L' , 'fusiform R' ,
                'inferiorparietal L' , 'inferiorparietal R' , 'inferiortemporal L' , 'inferiortemporal R' , 'insula L' ,
                'insula R' , 'isthmuscingulate L' , 'isthmuscingulate R' , 'lateraloccipital L' , 'lateraloccipital R' ,
                'lateralorbitofrontal L' , 'lateralorbitofrontal R' , 'lingual L' , 'lingual R' ,
                'medialorbitofrontal L' , 'medialorbitofrontal R' , 'middletemporal L' , 'middletemporal R' ,
                'paracentral L' , 'paracentral R' , 'parahippocampal L' , 'parahippocampal R' , 'parsopercularis L' ,
                'parsopercularis R' , 'parsorbitalis L' , 'parsorbitalis R' , 'parstriangularis L' ,
                'parstriangularis R' , 'pericalcarine L' , 'pericalcarine R' , 'postcentral L' , 'postcentral R' ,
                'posteriorcingulate L' , 'posteriorcingulate R' , 'precentral L' , 'precentral R' , 'precuneus L' ,
                'precuneus R' , 'rostralanteriorcingulate L' , 'rostralanteriorcingulate R' , 'rostralmiddlefrontal L' ,
                'rostralmiddlefrontal R' , 'superiorfrontal L' , 'superiorfrontal R' , 'superiorparietal L' ,
                'superiorparietal R' , 'superiortemporal L' , 'superiortemporal R' , 'supramarginal L' ,
                'supramarginal R' , 'temporalpole L' , 'temporalpole R' , 'transversetemporal L' ,
                'transversetemporal R' ]
ROIs = '%0d'.join ( ROI_DK_list )
Region_DK = [ 'LT' , 'RT' , 'LL' , 'RL' , 'LF' , 'RF' , 'LO' , 'RO' , 'LT' , 'RT' , 'LPF' , 'RPF' , 'LT' , 'RT' , 'LP' ,
              'RP' , 'LT' , 'RT' , 'LT' , 'RT' , 'LL' , 'RL' , 'LO' , 'RO' , 'LPF' , 'RPF' , 'LO' , 'RO' , 'LPF' ,
              'RPF' , 'LT' , 'RT' , 'LC' , 'RC' , 'LT' , 'RT' , 'LF' , 'RF' , 'LPF' , 'RPF' , 'LF' , 'RF' , 'LO' , 'RO' , 'LC' ,
              'RC' , 'LL' , 'RL' , 'LC' , 'RC' , 'LP' , 'RP' , 'LL' , 'RL' , 'LF' , 'RF' , 'LF' , 'RF' , 'LP' , 'RP' ,
              'LT' , 'RT' , 'LP' , 'RP' , 'LT' , 'RT' , 'LT' , 'RT' ] # partition provided by Brainstorm


idx_DK_postcent=[ i for i , j in enumerate ( ROI_DK_list ) if j == 'postcentral L' or  j == 'postcentral R']
idx_DK_Maudalmiddlefront=[ i for i , j in enumerate ( ROI_DK_list ) if j == 'caudalmiddlefrontal L' or  j == 'caudalmiddlefrontal R']
for i in idx_DK_postcent:
    Region_DK[i]='P'
for i in idx_DK_Maudalmiddlefront:
    Region_DK[i]='M'

# partition into lobes - DK
nb_ROIs_DK=len(Region_DK)
Lobes_DK = [ "" for x in range ( len ( Region_DK ) ) ]
for kk_roi in range ( len ( Region_DK ) ):
    Lobes_DK[ kk_roi ] = regions_bst_to_lobes ( Region_DK[ kk_roi ] )

idx_F_DK = [ i for i , j in enumerate ( Lobes_DK ) if j == 'F' ]
idx_M_DK = [ i for i , j in enumerate ( Lobes_DK ) if j == 'M' ]
idx_P_DK = [ i for i , j in enumerate ( Lobes_DK ) if j == 'P' ]
idx_O_DK = [ i for i , j in enumerate ( Lobes_DK ) if j == 'O' ]
idx_T_DK = [ i for i , j in enumerate ( Lobes_DK ) if j == 'T' ]
idx_lobes_DK = [ idx_F_DK , idx_M_DK , idx_T_DK , idx_P_DK , idx_O_DK ]

ROI_DK_list_F = [ROI_DK_list[i] for i in idx_F_DK]
ROI_DK_list_M = [ROI_DK_list[i] for i in idx_M_DK]
ROI_DK_list_T = [ROI_DK_list[i] for i in idx_T_DK]
ROI_DK_list_P = [ROI_DK_list[i] for i in idx_P_DK]
ROI_DK_list_O = [ROI_DK_list[i] for i in idx_O_DK]
ROI_DK_list_by_lobes = list(
    np.concatenate((ROI_DK_list_F, ROI_DK_list_M, ROI_DK_list_T, ROI_DK_list_P, ROI_DK_list_O)))

Region_DK_colors = Region_DK
for i in idx_F_DK:
    Region_DK_colors[i] = 'firebrick'
for i in idx_M_DK:
    Region_DK_colors[i] = 'darkorange'
for i in idx_T_DK:
    Region_DK_colors[i] = 'darkolivegreen'
for i in idx_P_DK:
    Region_DK_colors[i] = 'cadetblue'
for i in idx_O_DK:
    Region_DK_colors[i] = 'mediumpurple'

Region_DK_color_F = [Region_DK_colors[i] for i in idx_F_DK]
Region_DK_color_M = [Region_DK_colors[i] for i in idx_M_DK]
Region_DK_color_T = [Region_DK_colors[i] for i in idx_T_DK]
Region_DK_color_P = [Region_DK_colors[i] for i in idx_P_DK]
Region_DK_color_O = [Region_DK_colors[i] for i in idx_O_DK]
Region_DK_color_by_lobes = list(np.concatenate(
    (Region_DK_color_F, Region_DK_color_M, Region_DK_color_T, Region_DK_color_P, Region_DK_color_O)))

node_names = ROI_DK_list_by_lobes
node_order = list(np.hstack((ROI_DK_list_by_lobes[16:68][:], ROI_DK_list_by_lobes[0:16][:])))
node_angles = circular_layout(node_names=node_names, node_order=node_order,
                              start_pos=90
                              )
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
freqbands = {'paper': [3, 40]}

opt_trial_duration = [120, 300]#[30, 180]#[1, 30, 180]#, 100864/256] # one single trial for everything - in seconds
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

path_figures_root = "/Users/marieconstance.corsi/Documents/GitHub/Fenicotteri-equilibristi/Figures/Classification/"
labels = pd.read_csv(path_csv_root + "LabelsDesikanAtlas.csv")
index_values = labels.values
column_values = labels.values

results_atm = pd.DataFrame()
#results_plv = pd.DataFrame()
num_perm = 100
max_time = 100864/256 # recording length
ratio = 0.5 # overlap

df_Avg_ATM_HC_edges = dict()
df_Avg_ATM_HC_edges_nodal = dict()
df_Avg_ATM_EP1_edges = dict()
df_Avg_ATM_EP1_edges_nodal = dict()

for f in freqbands:
    fmin = freqbands[f][0]
    fmax = freqbands[f][1]
    epochs2use = epochs_concat[f].drop([2, 31, 32, 33, 34],
                                       reason='USER')  # to have the same nb of pz/subjects and because data from the second patient may be corrupted
    for kk_trial_crop in opt_trial_duration:
        nb_crops = int(math.floor(int(max_time)/kk_trial_crop)/ratio) # cf overlap 50%
        nb_subj = len(epochs2use.events)
        nb_ROIs = len(epochs2use.ch_names)


        temp_Avg_ATM_HC_edges = list()
        temp_Avg_ATM_HC_edges_nodal = list()
        temp_Avg_ATM_EP1_edges = list()
        temp_Avg_ATM_EP1_edges_nodal = list()

        for kk_crop in range(nb_crops):
            tmin = kk_trial_crop*(kk_crop)*ratio
            tmax = tmin + kk_trial_crop
            epochs_crop = epochs2use.copy().crop(tmin=tmin, tmax=tmax, include_tmax=True) # copy to avoid crop of the original epochs

            # %% Average values - ATM
            epochs_in_permuted_order = epochs_crop
            data = epochs_crop.get_data()
            nb_ROIs = np.shape(data)[1]

            temp = np.transpose(data, (1, 0, 2))
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

            ATM = np.zeros((nb_subj, nb_ROIs, nb_ROIs))
            # ATM computation with edge-wise optimization parameters
            for kk_trial in range(nb_subj):
                list_avalanches, min_size_avalanches, max_size_avalanches, list_avalanches_bysize, temp_ATM = find_avalanches(
                    zscored_data_ep[kk_trial, :, :], thresh=kk_zthresh_edge, val_duration=kk_val_duration_edge)
                # ATM: nb_subj x nb_ROIs x nb_ROIs matrix
                ATM[kk_trial, :, :] = temp_ATM

            fixed_reshape_ATM = np.nan_to_num(ATM, nan=0)  # replace nan by 0

            ATM_HC = fixed_reshape_ATM[:32, :, :]
            ATM_EP1 = fixed_reshape_ATM[31:nb_subj, :, :]

            temp_Avg_ATM_HC_edges.append(np.mean(ATM_HC, 0))
            temp_Avg_ATM_HC_edges_nodal.append(np.sum(temp_Avg_ATM_HC_edges, 0))
            temp_Avg_ATM_EP1_edges.append(np.mean(ATM_EP1, 0))
            temp_Avg_ATM_EP1_edges_nodal.append(np.sum(temp_Avg_ATM_EP1_edges, 0))

            # %% Features importance - ATM + SVM
            for iperm in range(n_perms): # in case we want to do some permutations to increase the statistical power
                perm = rng.permutation(len(epochs2use))
                epochs_in_permuted_order = epochs2use[perm]
                data_shuffle = epochs_in_permuted_order.get_data()
                label_shuffle = [labels_concat[f][x] for x in perm]
                nb_subj = len(data_shuffle)  # nb subjects/ pz here
                nb_ROIs = np.shape(data_shuffle)[1]

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
                ATM = np.empty((nb_subj, nb_ROIs, nb_ROIs))
                for kk_trial in range(nb_subj):
                    list_avalanches, min_size_avalanches, max_size_avalanches, list_avalanches_bysize, temp_ATM = find_avalanches(
                        zscored_data_ep[kk_trial, :, :], thresh=kk_zthresh_edge, val_duration=kk_val_duration_edge)
                    # ATM: nb_subj x nb_ROIs x nb_ROIs matrix
                    ATM[kk_trial, :, :] = temp_ATM
                reshape_ATM = np.reshape(ATM, (np.shape(ATM)[0], np.shape(ATM)[1] * np.shape(ATM)[2]))
                fixed_reshape_ATM_cl = np.nan_to_num(reshape_ATM, nan=0)

                # ATM computation with node-wise optimization parameters
                ATM_nodal = np.empty((nb_subj, nb_ROIs, nb_ROIs))
                for kk_trial in range(nb_subj):
                    list_avalanches, min_size_avalanches, max_size_avalanches, list_avalanches_bysize, temp_ATM_nodal = find_avalanches(
                        zscored_data_ep[kk_trial, :, :], thresh=kk_zthresh_nodal, val_duration=kk_val_duration_nodal)
                    # ATM: nb_subj x nb_ROIs x nb_ROIs matrix
                    ATM_nodal[kk_trial, :, :] = temp_ATM_nodal
                temp_ATM_nod = np.nan_to_num(ATM_nodal, nan=0)
                ATM_nodal_cl = np.sum(temp_ATM_nod, 1)

                # here just to find out the best hyperparameters, nothing more - need to be linear to retrieve features
                clf_prelim = GridSearchCV(SVC(kernel="linear"), {"C": [0.1, 1, 10]}, cv=5)
                clf_prelim.fit(fixed_reshape_ATM_cl, label_shuffle)
                best_C_ATM = clf_prelim.best_params_["C"]

                clf_prelim.fit(ATM_nodal_cl, label_shuffle)
                best_C_ATM_nodal = clf_prelim.best_params_["C"]

                # real classification now
                my_svm_ATM = SVC(C=best_C_ATM, kernel="linear")
                my_svm_ATM_nodal = SVC(C=best_C_ATM_nodal, kernel="linear")

                clf_ATM = Pipeline([('SVM', my_svm_ATM)])
                clf_ATM_nodal = Pipeline([('SVM', my_svm_ATM_nodal)])

                scores_ATM_SVM = cross_validate(clf_ATM, fixed_reshape_ATM_cl, label_shuffle, cv=cv, n_jobs=None,
                                                scoring=scoring,
                                                return_estimator=True)  # returns the estimator objects for each cv split!
                scores_ATM_nodal_SVM = cross_validate(clf_ATM_nodal, ATM_nodal_cl, label_shuffle, cv=cv, n_jobs=None,
                                                      scoring=scoring,
                                                      return_estimator=True)  # returns the estimator objects for each cv split!

                # compute and store normalized weights for each crop and each trial duration
                path2store = path_csv_root + "/SVM/ATM_nodal_weights_HC_EP1_SVM_Classification" + "-state-" + str(
                    kk_trial_crop) + '-kk_crop-' + str(kk_crop) + '-nbSplit' + str(nbSplit) + "_states.csv"
                weights_estim_ATM_nodal = _compute_normalized_weights(scores_ATM_nodal_SVM, nbSplit, path2store)

                path2store = path_csv_root + "/SVM/ATM_edges_weights_HC_EP1_SVM_Classification" + "-state-" + str(
                    kk_trial_crop) + '-kk_crop-' + str(kk_crop) + '-nbSplit' + str(nbSplit) + "_states.csv"
                weights_estim_ATM = _compute_normalized_weights(scores_ATM_SVM, nbSplit, path2store)

                # concatenate ATM results in a dedicated dataframe
                temp_results_atm_nodal = _build_df_atm_from_scores(scores=scores_ATM_nodal_SVM,
                                                                   ppl_name="ATM+SVM-nodal",
                                                                   nbSplit=nbSplit, zthresh_value=kk_zthresh_nodal,
                                                                   aval_duration_value=kk_val_duration_nodal, fmin=fmin,
                                                                   fmax=fmax, duration=kk_trial_crop)
                temp_results_atm = _build_df_atm_from_scores(scores=scores_ATM_SVM, ppl_name="ATM+SVM",
                                                             nbSplit=nbSplit, zthresh_value=kk_zthresh_edge,
                                                             aval_duration_value=kk_val_duration_edge, fmin=fmin,
                                                             fmax=fmax, duration=kk_trial_crop)
                results_atm = pd.concat((results_atm, temp_results_atm, temp_results_atm_nodal))

        Avg_ATM_HC_edges = np.mean(temp_Avg_ATM_HC_edges, 0)
        Avg_ATM_HC_edges_nodal = np.mean(temp_Avg_ATM_HC_edges_nodal, 0)
        Avg_ATM_EP1_edges = np.mean(temp_Avg_ATM_EP1_edges, 0)
        Avg_ATM_EP1_edges_nodal = np.mean(temp_Avg_ATM_EP1_edges_nodal, 0)

        df_Avg_ATM_HC_edges[kk_trial_crop] = pd.DataFrame(data=Avg_ATM_HC_edges, index=index_values, columns=column_values)
        df_Avg_ATM_HC_edges_nodal[kk_trial_crop] = pd.DataFrame(data=Avg_ATM_HC_edges_nodal)
        df_Avg_ATM_EP1_edges[kk_trial_crop] = pd.DataFrame(data=Avg_ATM_EP1_edges, index=index_values, columns=column_values)
        df_Avg_ATM_EP1_edges_nodal[kk_trial_crop] = pd.DataFrame(data=Avg_ATM_EP1_edges_nodal)

        # plot connectome
        [SubMatrix_F_F, SubMatrix_F_M, SubMatrix_F_T, SubMatrix_F_P, SubMatrix_F_O, \
         SubMatrix_M_M, SubMatrix_M_T, SubMatrix_M_P, SubMatrix_M_O, \
         SubMatrix_T_T, SubMatrix_T_P, SubMatrix_T_O, \
         SubMatrix_P_P, SubMatrix_P_O, \
         SubMatrix_O_O, Avg_ATM_HC_edges_2plot] = Lobes_Partition(Avg_ATM_HC_edges, idx_F_DK, idx_M_DK, idx_P_DK,
                                                                  idx_T_DK, idx_O_DK)

        fig_edges_HC, ax = plt.subplots(facecolor='white',
                                        subplot_kw=dict(polar=True))
        _plot_connectivity_circle(
            con=Avg_ATM_HC_edges_2plot, node_names=node_names, node_colors=Region_DK_color_by_lobes,
            colorbar=True,
            fontsize_names=7, fontsize_title=9,
            node_angles=node_angles,
            colormap='Greys', facecolor='white', textcolor='black',
            node_edgecolor='white',
            fig=fig_edges_HC)
        fig_edges_HC.savefig(
            path_figures_root + 'Connect_Avg_HC_Edges_ATM_fmin_' + str(fmin) + '_fmax_' + str(fmax) + "-state-" + str(
                    kk_trial_crop) + '-kk_crop-' + str(kk_crop) + '-nbSplit' + str(nbSplit) + "_states.png",
            dpi=600,
            facecolor='white')

        # plot connectome
        [SubMatrix_F_F, SubMatrix_F_M, SubMatrix_F_T, SubMatrix_F_P, SubMatrix_F_O, \
         SubMatrix_M_M, SubMatrix_M_T, SubMatrix_M_P, SubMatrix_M_O, \
         SubMatrix_T_T, SubMatrix_T_P, SubMatrix_T_O, \
         SubMatrix_P_P, SubMatrix_P_O, \
         SubMatrix_O_O, Avg_ATM_EP1_edges_2plot] = Lobes_Partition(Avg_ATM_EP1_edges, idx_F_DK, idx_M_DK, idx_P_DK,
                                                                   idx_T_DK, idx_O_DK)

        fig_edges_EP1, ax = plt.subplots(facecolor='white',
                                         subplot_kw=dict(polar=True))
        _plot_connectivity_circle(
            con=Avg_ATM_EP1_edges_2plot, node_names=node_names, node_colors=Region_DK_color_by_lobes,
            colorbar=True,
            fontsize_names=7, fontsize_title=9,
            node_angles=node_angles,
            colormap='Greys', facecolor='white', textcolor='black',
            node_edgecolor='white',
            fig=fig_edges_EP1)
        fig_edges_EP1.savefig(
            path_figures_root + 'Connect_Avg_EP1_Edges_ATM_fmin_' + str(fmin) + '_fmax_' + str(fmax) + "-state-" + str(
                    kk_trial_crop) + '-kk_crop-' + str(kk_crop) + '-nbSplit' + str(nbSplit) + "_states.png",
            dpi=600,
            facecolor='white')

# # concatenate results in a single dataframe
# results_global = results_atm #pd.concat((results_atm, results_plv))
# results_global.to_csv(
#         path_csv_root + "/SVM/TrialDurationEffect_HC_EP1_IndivOpt_ATM__SVM_Classification-allnode-2class-" +
#                 '-nbSplit' + str(nbSplit) + "_states.csv"
#         )
# print(
#         "saved " +
#         path_csv_root + "/SVM/TrialDurationEffect_HC_EP1_IndivOpt_ATM__SVM_Classification-allnode-2class-" +
#         '-nbSplit' + str(nbSplit) + "_states.csv"
#         )


#%% List of labels - Desikan
# TODO: update
path_figures_root = "/Users/marieconstance.corsi/Documents/GitHub/Fenicotteri-equilibristi/Figures/Classification/"

labels = pd.read_csv(path_csv_root + "LabelsDesikanAtlas.csv")
index_values = labels.values
column_values = labels.values

df_weights_estim_ATM_nodal = dict()
df_weights_estim_ATM_edges = dict()
df_weights_estim_ATM_edges_nodal = dict()

test_weights_estim_ATM_edges = list()
test_weights_estim_ATM_edges_nodal = list()
test_weights_estim_ATM_nodal = list()

pd_median_weights_estim_ATM_edges = dict()
pd_median_weights_estim_ATM_edges_nodal = dict()
pd_median_weights_estim_ATM_nodal = dict()

opt_trial_duration=[30, 120, 180, 300]

for f in freqbands:
    fmin = freqbands[f][0]
    fmax = freqbands[f][1]

    for kk_trial_crop in opt_trial_duration:
        nb_crops = int(math.floor(int(max_time) / kk_trial_crop) / ratio)  # cf overlap 50%
        for kk_crop in range(nb_crops):
            path2retrieve = path_csv_root + "/SVM/ATM_edges_weights_HC_EP1_SVM_Classification" + "-state-" + str(
                kk_trial_crop) + '-kk_crop-' + str(kk_crop) + '-nbSplit' + str(nbSplit) + "_states.csv"

            temp3 = pd.read_csv(path2retrieve)
            temp_weights_estim_ATM_edges = np.array(temp3["median"]).reshape(68,68)
            test_weights_estim_ATM_edges.append(np.array(temp3["median"]))

            weights_estim_ATM_edges_nodal = np.median(temp_weights_estim_ATM_edges,1)
            test_weights_estim_ATM_edges_nodal.append(weights_estim_ATM_edges_nodal)

            path2retrieve = path_csv_root + "/SVM/ATM_nodal_weights_HC_EP1_SVM_Classification" + "-state-" + str(
                kk_trial_crop) + '-kk_crop-' + str(kk_crop) + '-nbSplit' + str(nbSplit) + "_states.csv"

            temp4 = pd.read_csv(path2retrieve)
            weights_estim_ATM_nodal = temp4["median"]
            test_weights_estim_ATM_nodal.append(weights_estim_ATM_nodal)

        pd_weights_estim_ATM_edges = pd.DataFrame(test_weights_estim_ATM_edges)
        pd_median_weights_estim_ATM_edges[kk_trial_crop]=pd_weights_estim_ATM_edges.median()
        pd_weights_estim_ATM_edges_nodal = pd.DataFrame(test_weights_estim_ATM_edges_nodal)
        pd_median_weights_estim_ATM_edges_nodal[kk_trial_crop]=pd_weights_estim_ATM_edges_nodal.median()
        pd_weights_estim_ATM_nodal = pd.DataFrame(test_weights_estim_ATM_nodal)
        pd_median_weights_estim_ATM_nodal[kk_trial_crop]=pd_weights_estim_ATM_nodal.median()

        # plot connectomes
        mat_median_edges = np.reshape(pd_median_weights_estim_ATM_edges[kk_trial_crop].values, (68,68))
        [SubMatrix_F_F, SubMatrix_F_M, SubMatrix_F_T, SubMatrix_F_P, SubMatrix_F_O, \
         SubMatrix_M_M, SubMatrix_M_T, SubMatrix_M_P, SubMatrix_M_O, \
         SubMatrix_T_T, SubMatrix_T_P, SubMatrix_T_O, \
         SubMatrix_P_P, SubMatrix_P_O, \
         SubMatrix_O_O, Avg_ATM_EP1_edges_2plot] = Lobes_Partition(mat_median_edges, idx_F_DK, idx_M_DK, idx_P_DK,
                                                                   idx_T_DK, idx_O_DK)

        fig_edges_EP1, ax = plt.subplots(facecolor='white',
                                         subplot_kw=dict(polar=True))
        _plot_connectivity_circle(
            con=Avg_ATM_EP1_edges_2plot, node_names=node_names, node_colors=Region_DK_color_by_lobes,
            colorbar=True,
            fontsize_names=7, fontsize_title=9,
            node_angles=node_angles,
            colormap='Greys', facecolor='white', textcolor='black',
            node_edgecolor='white',
            fig=fig_edges_EP1)
        fig_edges_EP1.savefig(
            path_figures_root + 'Connect_Features_EP1_Edges_ATM_fmin_' + str(fmin) + '_fmax_' + str(fmax) + "-state-" + str(
                kk_trial_crop) + '-kk_crop-' + str(kk_crop) + '-nbSplit' + str(nbSplit) + "_states.png",
            dpi=600,
            facecolor='white')

#%% plot connectome differences (avg) EP1 and HC
#opt_trial_duration=[120,300]
for f in freqbands:
    for kk_trial_crop in opt_trial_duration:
        EP1_edges = df_Avg_ATM_EP1_edges[kk_trial_crop].values
        HC_edges = df_Avg_ATM_HC_edges[kk_trial_crop].values
        Diff_edges = EP1_edges - HC_edges

        [SubMatrix_F_F, SubMatrix_F_M, SubMatrix_F_T, SubMatrix_F_P, SubMatrix_F_O, \
         SubMatrix_M_M, SubMatrix_M_T, SubMatrix_M_P, SubMatrix_M_O, \
         SubMatrix_T_T, SubMatrix_T_P, SubMatrix_T_O, \
         SubMatrix_P_P, SubMatrix_P_O, \
         SubMatrix_O_O, Avg_ATM_EP1_edges_2plot] = Lobes_Partition(Diff_edges, idx_F_DK, idx_M_DK, idx_P_DK,
                                                                   idx_T_DK, idx_O_DK)

        valmin=Diff_edges.min()
        valmax=Diff_edges.max()
        vmax= max(abs(valmin),abs(valmax))
        fig_edges_EP1, ax = plt.subplots(facecolor='white',
                                         subplot_kw=dict(polar=True))
        _plot_connectivity_circle(
            con=Avg_ATM_EP1_edges_2plot, node_names=node_names, node_colors=Region_DK_color_by_lobes,
            colorbar=True,
            fontsize_names=7, fontsize_title=9,
            node_angles=node_angles,
            colormap='RdBu_r', facecolor='white', textcolor='black',
            node_edgecolor='white',
            vmin=-vmax, vmax=vmax,
            fig=fig_edges_EP1)
        fig_edges_EP1.savefig(
            path_figures_root + 'Connect_Diff_Avg_EP1_HC_Edges_ATM_fmin_' + str(fmin) + '_fmax_' + str(fmax) + "-state-" + str(
                kk_trial_crop) + '-kk_crop-' + str(kk_crop) + '-nbSplit' + str(nbSplit) + "_states.png",
            dpi=600,
            facecolor='white')


# %% save .mat to plot nodal results into scalp
Avg_ATM_EP1_HC_states = {
    "df_Avg_ATM_EP1_edges_nodal_30s": df_Avg_ATM_EP1_edges_nodal[30].values,
    "df_Avg_ATM_EP1_edges_nodal_180s": df_Avg_ATM_EP1_edges_nodal[180].values,
    "df_Avg_ATM_HC_edges_nodal_30s": df_Avg_ATM_HC_edges_nodal[30].values,
    "df_Avg_ATM_HC_edges_nodal_180s": df_Avg_ATM_HC_edges_nodal[180].values,
}

scipy.io.savemat(path_csv_root+'Avg_ATM_EP1_HC_states.mat',Avg_ATM_EP1_HC_states)

Avg_ATM_EP1_HC_states = {
    "df_Avg_ATM_EP1_edges_nodal_120s": df_Avg_ATM_EP1_edges_nodal[120].values,
    "df_Avg_ATM_EP1_edges_nodal_300s": df_Avg_ATM_EP1_edges_nodal[300].values,
    "df_Avg_ATM_HC_edges_nodal_120s": df_Avg_ATM_HC_edges_nodal[120].values,
    "df_Avg_ATM_HC_edges_nodal_300s": df_Avg_ATM_HC_edges_nodal[300].values,
}

scipy.io.savemat(path_csv_root+'Avg_ATM_EP1_HC_states_out.mat',Avg_ATM_EP1_HC_states)

features = {
            "df_weights_estim_ATM_edges_nodal_30s": pd_median_weights_estim_ATM_edges_nodal[30].values,
            "df_weights_estim_ATM_nodal_30s": pd_median_weights_estim_ATM_nodal[30].values,
            "df_weights_estim_ATM_edges_30s": pd_median_weights_estim_ATM_edges[30].T.values,
            "df_weights_estim_ATM_edges_nodal_180s": pd_median_weights_estim_ATM_edges_nodal[180].values,
            "df_weights_estim_ATM_nodal_180s": pd_median_weights_estim_ATM_nodal[180].values,
            "df_weights_estim_ATM_edges_180s": pd_median_weights_estim_ATM_edges[180].T.values
            }
scipy.io.savemat(path_csv_root+'Features_ATM_states.mat',features)


features = {
            "df_weights_estim_ATM_edges_nodal_120s": pd_median_weights_estim_ATM_edges_nodal[120].values,
            "df_weights_estim_ATM_nodal_120s": pd_median_weights_estim_ATM_nodal[120].values,
            "df_weights_estim_ATM_edges_120s": pd_median_weights_estim_ATM_edges[120].T.values,
            "df_weights_estim_ATM_edges_nodal_300s": pd_median_weights_estim_ATM_edges_nodal[300].values,
            "df_weights_estim_ATM_nodal_300s": pd_median_weights_estim_ATM_nodal[300].values,
            "df_weights_estim_ATM_edges_300s": pd_median_weights_estim_ATM_edges[300].T.values,
            }
scipy.io.savemat(path_csv_root+'Features_ATM_states_out.mat',features)