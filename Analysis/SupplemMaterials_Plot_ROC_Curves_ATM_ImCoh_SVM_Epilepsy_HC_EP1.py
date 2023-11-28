# %%
"""
==============================================================
Attempt to plot ROC curves from ATM/ImCoh classification from MEG data in the source space
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
from mne_connectivity.viz import plot_connectivity_circle
from mne_connectivity.viz.circle import _plot_connectivity_circle
from mne.viz import circular_layout#, plot_connectivity_circle

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

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import ShuffleSplit, cross_val_score
from sklearn.model_selection import cross_validate
from sklearn import metrics, model_selection
from sklearn.metrics import DetCurveDisplay, RocCurveDisplay, accuracy_score, f1_score, roc_auc_score, precision_score, recall_score

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


#%% retrive optimal parameters for ATMs
opt_atm_param_edge = dict()
opt_atm_param_nodal = dict()
perf_opt_atm_param_edge = dict()
perf_opt_atm_param_nodal = dict()

df_res_opt_db=pd.read_csv(
    path_csv_root + "/SVM/OptConfig_HC_EP1_MEG_ATM_SVM_ClassificationRebuttal-allnode_rest_BroadBand.csv"
)
opt_atm_param_edge["paper"] = [df_res_opt_db["zthresh"][0], df_res_opt_db["val_duration"][0]]
perf_opt_atm_param_edge["paper"] = df_res_opt_db["test_accuracy"][0]

df_res_opt_theta_alpha=pd.read_csv(
    path_csv_root + "/SVM/OptConfig_HC_EP1_MEG_ATM_SVM_ClassificationRebuttal-allnode_rest_theta_alphaBand.csv"
)
opt_atm_param_edge["theta-alpha"] = [df_res_opt_theta_alpha["zthresh"][0], df_res_opt_theta_alpha["val_duration"][0]]
perf_opt_atm_param_edge["theta-alpha"] = df_res_opt_theta_alpha["test_accuracy"][0]

df_res_opt_theta=pd.read_csv(
    path_csv_root + "/SVM/OptConfig_HC_EP1_MEG_ATM_SVM_ClassificationRebuttal-allnode_rest_theta_Band.csv"
)
opt_atm_param_edge["theta"] = [df_res_opt_theta["zthresh"][0], df_res_opt_theta["val_duration"][0]]
perf_opt_atm_param_edge["theta"] = df_res_opt_theta["test_accuracy"][0]

df_res_opt_alpha=pd.read_csv(
    path_csv_root + "/SVM/OptConfig_HC_EP1_MEG_ATM_SVM_ClassificationRebuttal-allnode_rest_alphaBand.csv"
)
opt_atm_param_edge["alpha"] = [df_res_opt_alpha["zthresh"][0], df_res_opt_alpha["val_duration"][0]]
perf_opt_atm_param_edge["alpha"] = df_res_opt_alpha["test_accuracy"][0]


df_res_opt_db_nodal=pd.read_csv(
    path_csv_root + "/SVM/OptConfig_HC_EP1_MEG_ATM_SVM_ClassificationRebuttal-nodal_rest_BroadBand.csv"
)
opt_atm_param_nodal["paper"] = [df_res_opt_db_nodal["zthresh"][0], df_res_opt_db_nodal["val_duration"][0]]
perf_opt_atm_param_nodal["paper"] = df_res_opt_db_nodal["test_accuracy"][0]

df_res_opt_theta_alpha_nodal=pd.read_csv(
    path_csv_root + "/SVM/OptConfig_HC_EP1_MEG_ATM_SVM_ClassificationRebuttal-nodal_rest_theta_alphaBand.csv"
)
opt_atm_param_nodal["theta-alpha"] = [df_res_opt_theta_alpha_nodal["zthresh"][0], df_res_opt_theta_alpha_nodal["val_duration"][0]]
perf_opt_atm_param_nodal["theta-alpha"] = df_res_opt_theta_alpha_nodal["test_accuracy"][0]

df_res_opt_theta_nodal=pd.read_csv(
    path_csv_root + "/SVM/OptConfig_HC_EP1_MEG_ATM_SVM_ClassificationRebuttal-nodal_rest_theta_Band.csv"
)
opt_atm_param_nodal["theta"] = [df_res_opt_theta_nodal["zthresh"][0], df_res_opt_theta_nodal["val_duration"][0]]
perf_opt_atm_param_nodal["theta"] = df_res_opt_theta_nodal["test_accuracy"][0]

df_res_opt_alpha_nodal=pd.read_csv(
    path_csv_root + "/SVM/OptConfig_HC_EP1_MEG_ATM_SVM_ClassificationRebuttal-nodal_rest_alphaBand.csv"
)
opt_atm_param_nodal["alpha"] = [df_res_opt_alpha_nodal["zthresh"][0], df_res_opt_alpha_nodal["val_duration"][0]]
perf_opt_atm_param_nodal["alpha"] = df_res_opt_alpha_nodal["test_accuracy"][0]

#%% load data
opt_trial_duration = [100864 / 256]  # one single trial for everything - in seconds
fs = 256
freqbands = {'theta': [3, 8],
                 'alpha': [8, 14],
                 'theta-alpha': [3, 14],  # Epilepsy case
                 'paper': [3, 40]}

test = mat73.loadmat(
    '/Users/marieconstance.corsi/Documents/GitHub/Fenicotteri-equilibristi/Database/1_Clinical/Epilepsy_GMD/Data_Epi_MEG_4Classif_concat_NoTrials.mat')
ch_names = test['labels_AAL1']
ch_types = ["eeg" for i in range(np.shape(ch_names)[0])]


grp_id_2use = ['HC', 'EPI 1']
precomp_concat_name = path_data_root + '/concat_epochs_HC_EP1.gz'
path_figures_root = "/Users/marieconstance.corsi/Documents/GitHub/Fenicotteri-equilibristi/Figures/Classification/"

if osp.exists(precomp_concat_name):
    print("Loading existing concatenated precomputations...")
    with gzip.open(precomp_concat_name, "rb") as file:
        epochs_concat, labels_concat = pickle.load(file)
else:
    print("Please consider performing precomputations and/or concatenation of the epochs!")

labels = pd.read_csv(path_csv_root + "LabelsDesikanAtlas.csv")
index_values = labels.values
column_values = labels.values

#%% parameters classification
nbSplit = 50
scoring = ['precision', 'recall', 'accuracy', 'f1', 'roc_auc']
# shuffle order of epochs & labels
rng = np.random.default_rng(42)  # set a random seed
n_perms = 1#00  # number of permutations wanted

# parameters for the default classifier & cross-validation
svm = GridSearchCV(SVC(), {"kernel": ("linear", "rbf"), "C": [0.1, 1, 10]}, cv=5)
cv = ShuffleSplit(nbSplit, test_size=0.2, random_state=21)

#%%
epochs_concat['theta'] = epochs_concat['theta-alpha']
epochs_concat['alpha'] = epochs_concat['paper']
labels_concat['theta'] = labels_concat['theta-alpha']
labels_concat['alpha'] = labels_concat['paper']

for kk_trial_crop in opt_trial_duration:
    tmin = 0
    tmax = kk_trial_crop

    for f in freqbands:
        fmin = freqbands[f][0]
        fmax = freqbands[f][1]
        epochs2use = epochs_concat[f].drop([2, 31, 32, 33, 34],
                                           reason='USER')  # to have the same nb of pz/subjects and because data from the second patient may be corrupted
        epochs_crop = epochs2use.crop(tmin=tmin, tmax=tmax, include_tmax=True)
        n_perms = 1

        for iperm in range(n_perms):  # in case we want to do some permutations to increase the statistical power
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

                ATM[kk_trial, :, :] = temp_ATM

            clf = Pipeline([('SVM', svm)])
            reshape_ATM = np.reshape(ATM, (np.shape(ATM)[0], np.shape(ATM)[1] * np.shape(ATM)[2]))
            fixed_reshape_ATM = np.nan_to_num(reshape_ATM, nan=0)  # replace nan by 0
            temp_ATM = np.nan_to_num(ATM, nan=0)
            ATM_nodal = np.sum(temp_ATM, 1)

            # plot ROC Curve x ATM
            # node-wise classification
            X = ATM_nodal
            y = label_shuffle
            X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, # 0.25 for the ratio
                                                                                random_state=21)  # random state = like we did previously
            clf = Pipeline([('SVM', svm)])
            clf.fit(X_train, y_train)
            name='ATM_nodal+SVM, ' + f
            fig, [ax_roc, ax_det] = plt.subplots(1,2, figsize=(11,5))
            RocCurveDisplay.from_estimator(clf, X_test, y_test, ax=ax_roc, name=name)
            DetCurveDisplay.from_estimator(clf, X_test, y_test, ax=ax_det, name=name)
            ax_roc.set_title('Receiver Operating Characteristic (ROC) Curves')
            ax_det.set_title("Detection Error Tradeoff (DET curve)")
            ax_roc.grid(linestyle="--")
            ax_det.grid(linestyle="--")
            plt.legend()
            plt.savefig(path_figures_root+name+'_ROC_Det.pdf')

            # edge-wise classification
            X = fixed_reshape_ATM
            y = label_shuffle
            X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, 
                                                                                random_state=21)  # random state = like we did previously
            clf = Pipeline([('SVM', svm)])
            clf.fit(X_train, y_train)
            name='ATM_edge+SVM, ' + f
            fig, [ax_roc, ax_det] = plt.subplots(1,2, figsize=(11,5))
            RocCurveDisplay.from_estimator(clf, X_test, y_test, ax=ax_roc, name=name)
            DetCurveDisplay.from_estimator(clf, X_test, y_test, ax=ax_det, name=name)
            ax_roc.set_title('Receiver Operating Characteristic (ROC) Curves')
            ax_det.set_title("Detection Error Tradeoff (DET curve)")
            ax_roc.grid(linestyle="--")
            ax_det.grid(linestyle="--")
            plt.legend()
            plt.savefig(path_figures_root+name+'_ROC_Det.pdf')


            # ImCoh
            ft = FunctionalTransformer(
                delta=10, ratio=0.5, method="imcoh", fmin=fmin, fmax=fmax
            )
            preproc_meg = Pipeline(steps=[("ft", ft)])  # , ("spd", EnsureSPD())])
            mat_ImCoh = preproc_meg.fit_transform(
                epochs_in_permuted_order)  # actually unfair because does an average of the estimations ie more robust than ATM this way

            ImCoh_cl = np.reshape(mat_ImCoh, (nb_trials, nb_ROIs * nb_ROIs))
            ImCoh_nodal_cl = np.sum(mat_ImCoh, 1)

            # node-wise classification
            X = ImCoh_nodal_cl
            y = label_shuffle
            X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y,  random_state=21) # random state = like we did previously
            clf = Pipeline([('SVM', svm)])
            clf.fit(X_train, y_train)
            name='ImCoh_nodal+SVM, ' + f
            fig, [ax_roc, ax_det] = plt.subplots(1,2, figsize=(11,5))
            RocCurveDisplay.from_estimator(clf, X_test, y_test, ax=ax_roc, name=name)
            DetCurveDisplay.from_estimator(clf, X_test, y_test, ax=ax_det, name=name)
            ax_roc.set_title('Receiver Operating Characteristic (ROC) Curves')
            ax_det.set_title("Detection Error Tradeoff (DET curve)")
            ax_roc.grid(linestyle="--")
            ax_det.grid(linestyle="--")
            plt.legend()
            plt.savefig(path_figures_root+name+'_ROC_Det.pdf')

            # edge-wise classification
            X = ImCoh_cl
            y = label_shuffle
            X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y,  random_state=21) # random state = like we did previously
            clf = Pipeline([('SVM', svm)])
            clf.fit(X_train, y_train)
            name='ImCoh_edge+SVM, ' + f
            fig, [ax_roc, ax_det] = plt.subplots(1,2, figsize=(11,5))
            RocCurveDisplay.from_estimator(clf, X_test, y_test, ax=ax_roc, name=name)
            DetCurveDisplay.from_estimator(clf, X_test, y_test, ax=ax_det, name=name)
            ax_roc.set_title('Receiver Operating Characteristic (ROC) Curves')
            ax_det.set_title("Detection Error Tradeoff (DET curve)")
            ax_roc.grid(linestyle="--")
            ax_det.grid(linestyle="--")
            plt.legend()
            plt.savefig(path_figures_root+name+'_ROC_Det.pdf')