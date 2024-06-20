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
from sklearn.metrics import confusion_matrix, RocCurveDisplay, auc

import numpy as np
from scipy.stats import zscore
from moabb.paradigms import MotorImagery

# to compute ImCoh estimations for each trial
from Scripts.py_viz.fc_pipeline import FunctionalTransformer

import matplotlib.pyplot as plt
import seaborn as sns

# %%
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
scoring = confusion_matrix_scorer # ['precision', 'recall', 'accuracy', 'f1', 'roc_auc']
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
freqbands = {'paper': [3,40],
             'theta-alpha': [3,14],  # reboot
             'alpha-beta': [8, 30],  # requested by the reviewer
             'theta-alpha-beta': [3, 30],  # requested by the reviewer
             'beta-gamma': [14, 40],  # requested by the reviewer
             }

opt_trial_duration = [100864/256] # one single trial for everything
fs = 256

test=mat73.loadmat(path_data_root+'1_Clinical/Epilepsy_GMD/Data_Epi_MEG_4Classif_concat_NoTrials.mat')
ch_names = test['labels_AAL1']
ch_types = ["eeg" for i in range(np.shape(ch_names)[0])]

#%% retrieve optimal parameters, for each freq band [zthresh, aval_dur]
opt_atm_param_edge = dict()
opt_atm_param_nodal = dict()
perf_opt_atm_param_edge = dict()
perf_opt_atm_param_nodal = dict()

df_res_opt_db=pd.read_csv(
    path_csv_root + "/SVM/OptConfig_HC_EP1_MEG_ATM_SVM_Classification-allnode_rest_BroadBand.csv"
)
opt_atm_param_edge["paper"] = [df_res_opt_db["zthresh"][0], df_res_opt_db["val_duration"][0]]
perf_opt_atm_param_edge["paper"] = df_res_opt_db["test_accuracy"][0]

df_res_opt_theta_alpha=pd.read_csv(
    path_csv_root + "/SVM/OptConfig_HC_EP1_MEG_ATM_SVM_Classification-allnode_rest_theta_alpha_Band.csv"
)
opt_atm_param_edge["theta-alpha"] = [df_res_opt_theta_alpha["zthresh"][0], df_res_opt_theta_alpha["val_duration"][0]]
perf_opt_atm_param_edge["theta-alpha"] = df_res_opt_theta_alpha["test_accuracy"][0]

df_res_opt_alpha_beta=pd.read_csv(
    path_csv_root + "/SVM/OptConfig_HC_EP1_MEG_ATM_SVM_Classification-allnode_rest_alpha_beta_Band.csv"
)
opt_atm_param_edge["alpha-beta"] = [df_res_opt_alpha_beta["zthresh"][0], df_res_opt_alpha_beta["val_duration"][0]]
perf_opt_atm_param_edge["alpha-beta"] = df_res_opt_alpha_beta["test_accuracy"][0]

df_res_opt_beta_gamma=pd.read_csv(
    path_csv_root + "/SVM/OptConfig_HC_EP1_MEG_ATM_SVM_Classification-allnode_rest_beta_gamma_Band.csv"
)
opt_atm_param_edge["beta-gamma"] = [df_res_opt_beta_gamma["zthresh"][0], df_res_opt_beta_gamma["val_duration"][0]]
perf_opt_atm_param_edge["beta-gamma"] = df_res_opt_beta_gamma["test_accuracy"][0]

df_res_opt_theta_alpha_beta=pd.read_csv(
    path_csv_root + "/SVM/OptConfig_HC_EP1_MEG_ATM_SVM_Classification-allnode_rest_df_res_opt_theta_alpha_beta_Band.csv"
)
opt_atm_param_edge["theta-alpha-beta"] = [df_res_opt_theta_alpha_beta["zthresh"][0], df_res_opt_theta_alpha_beta["val_duration"][0]]
perf_opt_atm_param_edge["theta-alpha-beta"] = df_res_opt_theta_alpha_beta["test_accuracy"][0]

#%% test 1 - classification HC vs EP1 - 31 vs 31
grp_id_2use = ['HC', 'EPI 1']
#precomp_concat_name = path_data_root + '/concat_epochs_HC_EP1.gz'

results_atm = pd.DataFrame()
results_ImCoh = pd.DataFrame()
num_perm = 100
max_time = 100864/256 # recording length

# NEW - 21/04/2024 - to ensure to have ALWAYS the same order in the presentation of pz/hc - once for all now:
perm_idx_filename =  path_data_root + '/Permuted_Idx_Classification_Rebuttal.gz'
if osp.exists(perm_idx_filename):
    print("Loading existing permuted indices to be applied...")
    with gzip.open(perm_idx_filename, "rb") as file:
        perm = pickle.load(file)
else:
    print("Redo the previous analysis with the same permuted index file saved!")


for f in freqbands:
    precomp_concat_name_f = path_data_root + '/concat_epochs_HC_EP1_'+f+'.gz'
    if osp.exists(precomp_concat_name_f):
        print("Loading existing concatenated precomputations...")
        with gzip.open(precomp_concat_name_f, "rb") as file:
            epochs_concat, labels_concat = pickle.load(file)
    else:
        print("Please consider performing precomputations and/or concatenation of the epochs!")

    fmin = freqbands[f][0]
    fmax = freqbands[f][1]

    epochs2use=epochs_concat[f].drop([2, 31, 32, 33, 34], reason='USER') # to have the same nb of pz/subjects and because data from the second patient may be corrupted
    for iperm in range(n_perms): # in case we want to do some permutations to increase the statistical power
        #perm = rng.permutation(len(epochs2use))
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
        mat_ImCoh = preproc_meg.fit_transform(epochs_in_permuted_order) # actually unfair because does an average of the estimations ie more robust than ATM this way

        ImCoh_cl = np.reshape(mat_ImCoh, (nb_trials, nb_ROIs * nb_ROIs))

        clf_2 = Pipeline([('SVM', svm)])
        # score_ImCoh_SVM = cross_val_score(clf_2, ImCoh_cl, labels, cv=cv, n_jobs=None)
        scores_ImCoh_SVM = cross_validate(clf_2, ImCoh_cl,  label_shuffle, cv=cv, n_jobs=None, scoring=scoring, return_estimator=False) # returns the estimator objects for each cv split!

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
        ax.plot(
            mean_fpr,
            mean_tpr,
            color = 'b',
            label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
            lw=2,
            alpha=0.8
        )

        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        ax.fill_between(
            mean_fpr,
            tprs_lower,
            tprs_upper,
            color="grey",
            alpha=0.2,
            label=r"$\pm$ 1 std. dev.",
        )
        ax.set(
            xlim=[-0.05, 1.05],
            ylim=[-0.05, 1.05],
            xlabel="False Positive Rate",
            ylabel="True Positive Rate",
            title=f"Mean ROC curve with variability - ImCoh (edge-wise) \nMean ROC (AUC = %0.2f $\pm$ %0.2f)" % (
            mean_auc, std_auc),
        )
        ax.axis("square")
        #ax.legend(loc='center left', bbox_to_anchor=(1,0.5))
        ax.get_legend().remove()
        plt.savefig(path_figures_root + "ROC_Curves_Opt_HC_EP1_ImCoh_Edge_SVM_Classification_2class-nbSplits"+str(nbSplit)+"_"+f+"_MEG.pdf", dpi=300)


        # concatenate ImCoh results in a dedicated dataframe
        pd_ImCoh_SVM = pd.DataFrame.from_dict(scores_ImCoh_SVM)
        temp_results_ImCoh = pd_ImCoh_SVM
        ppl_ImCoh = ["ImCoh+SVM"] * len(pd_ImCoh_SVM)
        temp_results_ImCoh["pipeline"] = ppl_ImCoh
        temp_results_ImCoh["split"] = [nbSplit] * len(ppl_ImCoh)
        temp_results_ImCoh["freq"] = [str(fmin) + '-' + str(fmax)] * len(ppl_ImCoh)

        results_ImCoh = pd.concat((results_ImCoh, temp_results_ImCoh))


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

        scores_ATM_SVM = cross_validate(clf_2, fixed_reshape_ATM, label_shuffle, cv=cv, n_jobs=None, scoring=scoring, return_estimator=False)

        # plot ROC with cross-validation - for edges only
        tprs =[]
        aucs = []
        mean_fpr = np.linspace(0,1,100)
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
            interp_tpr=np.interp(mean_fpr, viz.fpr, viz.tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
            aucs.append(viz.roc_auc)
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        ax.plot(
            mean_fpr,
            mean_tpr,
            color = 'b',
            label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
            lw=2,
            alpha=0.8
        )

        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        ax.fill_between(
            mean_fpr,
            tprs_lower,
            tprs_upper,
            color="grey",
            alpha=0.2,
            label=r"$\pm$ 1 std. dev.",
        )
        ax.set(
            xlim=[-0.05, 1.05],
            ylim=[-0.05, 1.05],
            xlabel="False Positive Rate",
            ylabel="True Positive Rate",
            title=f"Mean ROC curve with variability - ATM (edge-wise) \nMean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
        )
        ax.axis("square")
        #ax.legend(loc='center left', bbox_to_anchor=(1,0.5))
        ax.get_legend().remove()
        plt.savefig(path_figures_root + "ROC_Curves_Opt_HC_EP1_ATM_Edge_SVM_Classification_2class-nbSplits"+str(nbSplit)+"_"+f+"_MEG.pdf", dpi=300)


        # concatenate ATM results in a dedicated dataframe
        pd_ATM_SVM = pd.DataFrame.from_dict(scores_ATM_SVM)
        temp_results_atm = pd_ATM_SVM
        ppl_atm = ["ATM+SVM"] * len(pd_ATM_SVM)
        temp_results_atm["pipeline"] = ppl_atm
        temp_results_atm["split"] = [nbSplit] * len(ppl_atm)
        temp_results_atm["zthresh"] = [k_zthresh_edge] * len(ppl_atm)
        temp_results_atm["val_duration"] = [k_val_duration_edge] * len(ppl_atm)
        temp_results_atm["freq"] = [str(fmin) + '-' + str(fmax)] * len(ppl_atm)

        results_atm = pd.concat((results_atm, temp_results_atm))

    # concatenate results in a single dataframe
    results_global = pd.concat((results_atm, results_ImCoh))
    results_global.to_csv(
            path_csv_root + "/SVM/Paper_Sensivity_Specificity_HC_EP1_IndivOpt_ATM_ImCoh_Comparison_SVM_Classification-allnode-2class-" +
            "-freq-" + str(fmin) + '-' + str(fmax) + '-nbSplit' + str(nbSplit) + ".csv"
        )
    print(
            "saved " +
            path_csv_root + "/SVM/Paper_Sensivity_Specificity_HC_EP1_IndivOpt_ATM_ImCoh_Comparison_SVM_Classification-allnode-2class-" +
            "-freq-" + str(fmin) + '-' + str(fmax) + '-nbSplit' + str(nbSplit) + ".csv"
        )


#%%
results=pd.DataFrame()
freqbands_tot = {'paper': [3,40],
             'theta-alpha': [3,14],  # reboot
             'alpha-beta': [8, 30],  # requested by the reviewer
             'beta-gamma': [14, 40],  # requested by the reviewer
             'theta-alpha-beta': [3, 30],  # requested by the reviewer
             }

#freqbands_tot ={'paper': [3, 40]} # already concatenate all info
for f in freqbands_tot:
    fmin = freqbands_tot[f][0]
    fmax = freqbands_tot[f][1]
    temp_results = pd.read_csv(         path_csv_root + "/SVM/Paper_Sensivity_Specificity_HC_EP1_IndivOpt_ATM_ImCoh_Comparison_SVM_Classification-allnode-2class-" +
            "-freq-" + str(fmin) + '-' + str(fmax) + '-nbSplit' + str(nbSplit) + ".csv"
             )

    results = pd.concat((results, temp_results))


#%%
ppl_2plot = ["ATM+SVM", "ImCoh+SVM"]
results2plot = results[results["pipeline"].isin(ppl_2plot)]

sns.set_style("white")# whitegrid
g = sns.catplot(y="test_sensitivity",
                x='pipeline',
                hue="pipeline",
                kind='swarm',  # swarm
                col="freq",
                col_wrap=2,
                # dodge=True,
                #row='zthresh',
                height=4, aspect=4,
                data=results2plot, palette="tab10")

plt.savefig(path_figures_root + "Sensitivity_Opt_HC_EP1_ATM_vs_ImCoh_SVM_Classification_2class-nbSplits"+str(nbSplit)+"_broad_MEG.pdf", dpi=300)



#plt.style.use("dark_background")
g = sns.catplot(y="test_specificity",
                x='pipeline',
                hue="pipeline",
                kind='swarm',  # swarm
                col="freq",
                col_wrap=2,
                # dodge=True,
                #row='zthresh',
                height=4, aspect=3,
                data=results2plot, palette="tab10")

plt.savefig(path_figures_root + "Specificity_Opt_HC_EP1_ATM_vs_ImCoh_SVM_Classification_2class-nbSplits"+str(nbSplit)+"_broad_MEG.pdf", dpi=300)

#%% infos to compare results
results2plot_ATM=results2plot[(results2plot["pipeline"]=="ATM+SVM") & (results2plot["freq"]=='3-40')]
results2plot_ATM["test_sensitivity"].mean()
results2plot_ATM["test_sensitivity"].std()
results2plot_ATM["test_specificity"].mean()
results2plot_ATM["test_specificity"].std()


#%% reboot plots  based on accuracy from confusion matrix
import ptitprince as pt

dx = results2plot["test_accuracy_from_cm"]
dy = results2plot["pipeline"]
df = results2plot
ort="h"
sigma = .2
cut = 0.
width = .6
orient = "h"

g = sns.FacetGrid(df, col = "freq", height = 5)
g = g.map_dataframe(pt.RainCloud, x = dy, y = dx, data = df, bw = sigma,
                    orient = "h", palette='flare')
g.fig.subplots_adjust(top=0.75)
g.savefig(path_figures_root+'AccuracyResults_RaincloudPlots_FreqBands.png', dpi=600, facecolor='white')
g.savefig(path_figures_root+'Fig1A_FreqBands.eps', format='eps', dpi=600, facecolor='white')

#################################################################################
#%% plot ROC Curves like in the paper
grp_id_2use = ['HC', 'EPI 1']

results_atm = pd.DataFrame()
results_ImCoh = pd.DataFrame()
num_perm = 100
max_time = 100864/256 # recording length

# NEW - 21/04/2024 - to ensure to have ALWAYS the same order in the presentation of pz/hc - once for all now:
perm_idx_filename =  path_data_root + '/Permuted_Idx_Classification_Rebuttal.gz'
if osp.exists(perm_idx_filename):
    print("Loading existing permuted indices to be applied...")
    with gzip.open(perm_idx_filename, "rb") as file:
        perm = pickle.load(file)
else:
    print("Redo the previous analysis with the same permuted index file saved!")

for f in freqbands:
    precomp_concat_name_f = path_data_root + '/concat_epochs_HC_EP1_'+f+'.gz'
    if osp.exists(precomp_concat_name_f):
        print("Loading existing concatenated precomputations...")
        with gzip.open(precomp_concat_name_f, "rb") as file:
            epochs_concat, labels_concat = pickle.load(file)
    else:
        print("Please consider performing precomputations and/or concatenation of the epochs!")

    fmin = freqbands[f][0]
    fmax = freqbands[f][1]

    epochs2use=epochs_concat[f].drop([2, 31, 32, 33, 34], reason='USER') # to have the same nb of pz/subjects and because data from the second patient may be corrupted
    for iperm in range(n_perms): # in case we want to do some permutations to increase the statistical power
        #perm = rng.permutation(len(epochs2use))
        epochs_in_permuted_order = epochs2use[perm[0]]
        data_shuffle = epochs_in_permuted_order.get_data()
        label_shuffle = [labels_concat[f][x] for x in perm[0]]
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

        #score_ATM_SVM = cross_val_score(clf_2, reshape_ATM, labels, cv=cv, n_jobs=None)
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