# %%
"""
==============================================================
Plot ROC curves from ATM/ImCoh classification averaged across CV splits
===============================================================

"""
# Authors: Marie-Constance Corsi <marie-constance.corsi@inria.fr>
#
# License: BSD (3-clause)

import gzip

import mat73
import matplotlib.pyplot as plt

import numpy as np

import os.path as osp
import os
import pandas as pd
import pickle

from scipy.stats import zscore

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import ShuffleSplit, cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.metrics import confusion_matrix, RocCurveDisplay, auc

# to compute ImCoh estimations for each trial
from Analysis.fc_pipeline import FunctionalTransformer

# %% to be adapted
if os.path.basename(os.getcwd()) == "NeuronalAvalanches_TemporalLobeEpilepsy_EEG":
    os.chdir("Database/1_Clinical/Epilepsy_GMD/")
if os.path.basename(os.getcwd()) == "py_viz":
    os.chdir("/Users/marieconstance.corsi/Documents/GitHub/NeuronalAvalanches_TemporalLobeEpilepsy_EEG/Database/1_Clinical/Epilepsy_GMD")
basedir = os.getcwd()

path_csv_root = os.getcwd() + '/1_Dataset-csv/'
if not osp.exists(path_csv_root):
    os.mkdir(path_csv_root)
path_data_root = os.getcwd()
if not osp.exists(path_data_root):
    os.mkdir(path_data_root)
path_data_root_chan = os.getcwd()

path_figures_root = "/Users/marieconstance.corsi/Documents/GitHub/NeuronalAvalanches_TemporalLobeEpilepsy_EEG/Figures/Classification/"

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


def confusion_matrix_scorer(clf,X,y):
    y_pred = clf.predict(X)
    cm = confusion_matrix(y, y_pred)

    total1 = sum(sum(cm))
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn+fp)
    sensitivity = tp / (tp+fn)
    accuracy = (tp+tn)/total1

    return {'tn': cm[0,1], 'fp': cm[1,1],
            'fn': cm[0,1], 'tp': cm[1,1],
            'accuracy_from_cm': accuracy,
            'sensitivity': sensitivity,
            'specificity': specificity
            }
#%% parameters for classification
# k_components = 8
nbSplit = 50
scoring = confusion_matrix_scorer
# shuffle order of epochs & labels
rng = np.random.default_rng(42)  # set a random seed
n_perms = 1#00  # number of permutations wanted

# parameters for the default classifier & cross-validation
svm = GridSearchCV(SVC(), {"kernel": ("linear", "rbf"), "C": [0.1, 1, 10]}, cv=5)
cv = ShuffleSplit(nbSplit, test_size=0.2, random_state=21)

# %% parameters to be applied to extract the features
freqbands = {'theta-alpha': [3, 14],  # Epilepsy case
             'paper': [3, 40]}

opt_trial_duration = [100864/256] # one single trial for everything
fs = 256

test=mat73.loadmat('/Users/marieconstance.corsi/Documents/GitHub/NeuronalAvalanches_TemporalLobeEpilepsy_EEG/Database/1_Clinical/Epilepsy_GMD/Data_Epi_MEG_4Classif_concat_NoTrials.mat')
ch_names = test['labels_AAL1']
ch_types = ["eeg" for i in range(np.shape(ch_names)[0])]

#%% retrieve optimal parameters, for each freq band [zthresh, aval_dur]
opt_atm_param_edge = dict()
perf_opt_atm_param_edge = dict()

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

#%% classification HC vs EP1 - 31 vs 31
grp_id_2use = ['HC', 'EPI 1']
precomp_concat_name = path_data_root + '/concat_epochs_HC_EP1.gz'

if osp.exists(precomp_concat_name):
    print("Loading existing concatenated precomputations...")
    with gzip.open(precomp_concat_name, "rb") as file:
        epochs_concat, labels_concat = pickle.load(file)
else:
    print("Please consider performing precomputations and/or concatenation of the epochs!")

results_atm = pd.DataFrame()
results_ImCoh = pd.DataFrame()
num_perm = 100
max_time = 100864/256 # recording length
for f in freqbands:
    fmin = freqbands[f][0]
    fmax = freqbands[f][1]
    epochs2use = epochs_concat[f].drop([2, 31, 32, 33, 34],
                                       reason='USER')  # to have the same nb of pz/subjects and because data from the second patient may be corrupted
    for iperm in range(n_perms): # in case we want to do some permutations to increase the statistical power
        perm = rng.permutation(len(epochs2use))
        epochs_in_permuted_order = epochs2use[perm]
        data_shuffle = epochs_in_permuted_order.get_data()
        label_shuffle = [labels_concat[f][x] for x in perm]
        nb_trials = len(data_shuffle)  # nb subjects/ pz here
        nb_ROIs = np.shape(data_shuffle)[1]

        ### ImCoh + SVM
        ft = FunctionalTransformer(
            delta=30, ratio=0.5, method="imcoh", fmin=fmin, fmax=fmax
        )
        preproc_meg = Pipeline(steps=[("ft", ft)])
        mat_ImCoh = preproc_meg.fit_transform(epochs_in_permuted_order)

        ImCoh_cl = np.reshape(mat_ImCoh, (nb_trials, nb_ROIs * nb_ROIs))
        ImCoh_nodal_cl = np.sum(mat_ImCoh,1)

        clf_2 = Pipeline([('SVM', svm)])
        scores_ImCoh_SVM = cross_validate(clf_2, ImCoh_cl,  label_shuffle, cv=cv, n_jobs=None, scoring=scoring, return_estimator=False) # returns the estimator objects for each cv split!
        scores_ImCoh_nodal_SVM = cross_validate(clf_2, ImCoh_nodal_cl,  label_shuffle, cv=cv, n_jobs=None, scoring=scoring, return_estimator=False) # returns the estimator objects for each cv split!

        # plot ROC with cross-validation - for edges only
        tprs =[]
        aucs = []
        mean_fpr = np.linspace(0,1,100)
        fig, ax = plt.subplots(figsize=(6,6))
        for fold, (train, test) in enumerate(cv.split(ImCoh_cl,  label_shuffle)):
            X = [ImCoh_cl[index] for index in train]
            y = [label_shuffle[index] for index in train]
            clf_2.fit(X, y)
            X_test = [ImCoh_cl[index] for index in test]
            y_test = [label_shuffle[index] for index in test]
            viz = RocCurveDisplay.from_estimator(
                clf_2, X_test, y_test, name=f'ROC fold {fold}',
                alpha=0.3, lw=1, ax=ax, plot_chance_level=(fold==nbSplit-1)
            )
            interp_tpr=np.interp(mean_fpr, viz.fpr, viz.tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
            aucs.append(viz.roc_auc)
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)

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


        ### ATM + SVM -- attempt to optimize the code to make it faster...
        # get optimal parameters to make the process faster
        k_zthresh_edge = opt_atm_param_edge[f][0]
        k_val_duration_edge = opt_atm_param_edge[f][1]

        temp = np.transpose(data_shuffle, (1, 0, 2))
        temp_nc = np.reshape(temp, (np.shape(temp)[0], np.shape(temp)[1] * np.shape(temp)[2]))
        zscored_data = zscore(temp_nc, axis=1)
        # epoching here before computing the avalanches
        temp_zscored_data_ep = np.reshape(zscored_data,
                                          (np.shape(temp)[0], np.shape(temp)[1], np.shape(temp)[2]))
        zscored_data_ep = np.transpose(temp_zscored_data_ep, (1, 0, 2))

        ATM = np.empty((nb_trials, nb_ROIs, nb_ROIs))
        for k_trial in range(nb_trials):
            list_avalanches, min_size_avalanches, max_size_avalanches, list_avalanches_bysize, temp_ATM = find_avalanches(
                zscored_data_ep[k_trial, :, :], thresh=k_zthresh_edge, val_duration=k_val_duration_edge)
            # ATM: nb_trials x nb_ROIs x nb_ROIs matrix
            ATM[k_trial, :, :] = temp_ATM

        clf_2 = Pipeline([('SVM', svm)])
        reshape_ATM = np.reshape(ATM, (np.shape(ATM)[0], np.shape(ATM)[1] * np.shape(ATM)[2]))
        fixed_reshape_ATM = np.nan_to_num(reshape_ATM, nan=0)  # replace nan by 0
        temp_ATM = np.nan_to_num(ATM, nan=0)
        ATM_nodal = np.sum(temp_ATM, 1)

        scores_ATM_SVM = cross_validate(clf_2, fixed_reshape_ATM, label_shuffle, cv=cv, n_jobs=None, scoring=scoring, return_estimator=False)
        scores_ATM_SVM_nodal = cross_validate(clf_2, ATM_nodal, label_shuffle, cv=cv, n_jobs=None,
                                        scoring=scoring, return_estimator=False)


        # plot ROC with cross-validation - for edges only
        tprs_atm =[]
        aucs_atm = []
        mean_fpr_atm = np.linspace(0,1,100)
        fig, ax = plt.subplots(figsize=(6,6))
        for fold, (train, test) in enumerate(cv.split(fixed_reshape_ATM,  label_shuffle)):
            X = [fixed_reshape_ATM[index] for index in train]
            y = [label_shuffle[index] for index in train]
            clf_2.fit(X, y)
            X_test = [fixed_reshape_ATM[index] for index in test]
            y_test = [label_shuffle[index] for index in test]
            viz = RocCurveDisplay.from_estimator(
                clf_2, X_test, y_test, name=f'ROC fold {fold}',
                alpha=0.3, lw=1, ax=ax, plot_chance_level=(fold==nbSplit-1)
            )
            interp_tpr_atm=np.interp(mean_fpr_atm, viz.fpr, viz.tpr)
            interp_tpr_atm[0] = 0.0
            tprs_atm.append(interp_tpr_atm)
            aucs_atm.append(viz.roc_auc)
        mean_tpr_atm = np.mean(tprs_atm, axis=0)
        mean_tpr_atm[-1] = 1.0
        mean_auc_atm = auc(mean_fpr_atm, mean_tpr_atm)
        std_auc_atm = np.std(aucs_atm)

        plt.close('all')
        cmap = plt.cm.get_cmap('flare')
        fig1, ax1 = plt.subplots(figsize=(6, 6))
        line1=plt.plot(
            mean_fpr,
            mean_tpr,
            color = cmap(255),
            label=r"Mean ROC ImCoh (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
            lw=3,
            alpha=0.9
        )
        line2=plt.plot(
            mean_fpr,
            mean_tpr_atm,
            color = cmap(50),
            label=r"Mean ROC ATM (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc_atm, std_auc_atm),
            lw=3,
            alpha=1
        )
        ax1.set(
            xlim=[-0.05, 1.05],
            ylim=[-0.05, 1.05],
            xlabel="False Positive Rate",
            ylabel="True Positive Rate",
        )
        ax1.axis("square")
        plt.legend()
        plt.savefig(path_figures_root + "MeanROC_Curves_Opt_HC_EP1_ATM_ImCoh_Edge_SVM_Classification_2class-nbSplits"+str(nbSplit)+"_"+f+".pdf", dpi=300)
        plt.savefig(path_figures_root + 'Fig1B_'+f+".eps", format='eps', dpi=600, facecolor='white')