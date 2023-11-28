# %%
"""
==============================================================
Plot results comparison ImCoh vs ATM to classify HC vs EP1
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

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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

#%%
nbSplit = 50
scoring = ['precision', 'recall', 'accuracy', 'f1', 'roc_auc']
# recall = tp / (tp + fn) # default avg binary, to be changed if unbalanced/more classes - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html#sklearn.metrics.recall_score
# precision = tp / (tp + fp) # default avg binary, to be changed if unbalanced/more classes - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html#sklearn.metrics.precision_score
# f1 = F1 = 2 * (precision * recall) / (precision + recall) # here binary & balanced, otherwise need to be changed - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score
# roc_auc - Area under the receiver operating characteristic curve from prediction scores, default macro, to be changed for unbalanced and more classes - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html#sklearn.metrics.roc_auc_score


results=pd.DataFrame()
freqbands = {'theta': [3, 8],
                 'alpha': [8, 14],
                 'theta-alpha': [3, 14],  # Epilepsy case
                 'paper': [3, 40]}

for f in freqbands:
    fmin = freqbands[f][0]
    fmax = freqbands[f][1]
    temp_results = pd.read_csv(         path_csv_root + "/SVM/HC_EP1_IndivOpt_ATM_PLV_Comparison_SVM_Classification-allnode-2class-" +
            "-freq-" + str(fmin) + '-' + str(fmax) + '-nbSplit' + str(nbSplit) + ".csv"
             )

    results = pd.concat((results, temp_results))


#%%

path_figures_root = "/Users/marieconstance.corsi/Documents/GitHub/Fenicotteri-equilibristi/Figures/Classification/"

results_theta_alpha=results[results["freq"]=='3-14']
results_theta_alpha_atm=results_theta_alpha[results_theta_alpha["pipeline"]=="ATM+SVM"]
results_theta_alpha_atm_nodal=results_theta_alpha[results_theta_alpha["pipeline"]=="ATM+SVM-nodal"]

results_theta=results[results["freq"]=='3-8']
results_theta_atm=results_theta[results_theta["pipeline"]=="ATM+SVM"]
results_theta_atm_nodal=results_theta[results_theta["pipeline"]=="ATM+SVM-nodal"]

results_alpha=results[results["freq"]=='8-14']
results_alpha_atm=results_alpha[results_alpha["pipeline"]=="ATM+SVM"]
results_alpha_atm_nodal=results_alpha[results_alpha["pipeline"]=="ATM+SVM-nodal"]

results_broad=results[results["freq"]=='3-40']
results_broad_atm=results_broad[results_broad["pipeline"]=="ATM+SVM"]
results_broad_atm_nodal=results_broad[results_broad["pipeline"]=="ATM+SVM-nodal"]

#%% retrieve ATM results from optimization
df_res_opt_db = pd.read_csv(
    path_csv_root + "/SVM/OptConfig_HC_EP1_MEG_ATM_SVM_ClassificationRebuttal-allnode_rest_BroadBand.csv"
)
df_res_opt_db_median = pd.read_csv(
    path_csv_root + "/SVM/Median_OptConfig_HC_EP1_MEG_ATM_SVM_ClassificationRebuttal-allnode_rest_BroadBand.csv"
)
df_res_opt_theta_alpha = pd.read_csv(
    path_csv_root + "/SVM/OptConfig_HC_EP1_MEG_ATM_SVM_ClassificationRebuttal-allnode_rest_theta_alphaBand.csv"
)
df_res_opt_theta_alpha_median = pd.read_csv(
    path_csv_root + "/SVM/Median_OptConfig_HC_EP1_MEG_ATM_SVM_ClassificationRebuttal-allnode_rest_theta_alphaBand.csv"
)

df_res_opt_theta = pd.read_csv(
    path_csv_root + "/SVM/OptConfig_HC_EP1_MEG_ATM_SVM_ClassificationRebuttal-allnode_rest_theta_Band.csv"
)
df_res_opt_theta_median = pd.read_csv(
    path_csv_root + "/SVM/Median_OptConfig_HC_EP1_MEG_ATM_SVM_ClassificationRebuttal-allnode_rest_theta_Band.csv"
)
df_res_opt_alpha = pd.read_csv(
    path_csv_root + "/SVM/OptConfig_HC_EP1_MEG_ATM_SVM_ClassificationRebuttal-allnode_rest_alphaBand.csv"
)
df_res_opt_alpha_median = pd.read_csv(
    path_csv_root + "/SVM/Median_OptConfig_HC_EP1_MEG_ATM_SVM_ClassificationRebuttal-allnode_rest_alphaBand.csv"
)

df_res_opt_db_nodal = pd.read_csv(
    path_csv_root + "/SVM/OptConfig_HC_EP1_MEG_ATM_SVM_ClassificationRebuttal-nodal_rest_BroadBand.csv"
)
df_res_opt_db_median_nodal = pd.read_csv(
    path_csv_root + "/SVM/Median_OptConfig_HC_EP1_MEG_ATM_SVM_ClassificationRebuttal-nodal_rest_BroadBand.csv"
)
df_res_opt_theta_alpha_nodal = pd.read_csv(
    path_csv_root + "/SVM/OptConfig_HC_EP1_MEG_ATM_SVM_ClassificationRebuttal-nodal_rest_theta_alphaBand.csv"
)
df_res_opt_theta_alpha_median_nodal = pd.read_csv(
    path_csv_root + "/SVM/Median_OptConfig_HC_EP1_MEG_ATM_SVM_ClassificationRebuttal-nodal_rest_theta_alphaBand.csv"
)
df_res_opt_theta_nodal = pd.read_csv(
    path_csv_root + "/SVM/OptConfig_HC_EP1_MEG_ATM_SVM_ClassificationRebuttal-nodal_rest_theta_Band.csv"
)
df_res_opt_theta_median_nodal = pd.read_csv(
    path_csv_root + "/SVM/Median_OptConfig_HC_EP1_MEG_ATM_SVM_ClassificationRebuttal-nodal_rest_theta_Band.csv"
)
df_res_opt_alpha_nodal = pd.read_csv(
    path_csv_root + "/SVM/OptConfig_HC_EP1_MEG_ATM_SVM_ClassificationRebuttal-nodal_rest_alphaBand.csv"
)
df_res_opt_alpha_median_nodal = pd.read_csv(
    path_csv_root + "/SVM/Median_OptConfig_HC_EP1_MEG_ATM_SVM_ClassificationRebuttal-nodal_rest_alphaBand.csv"
)

#%% retrieve ImCoh results
results_ImCoh_alone=pd.DataFrame()


temp_results = pd.read_csv(         path_csv_root + "/SVM/Tril_HC_EP1_IndivOpt_ImCoh_SVM_Classification-allnode-2class-" +
            "-freq-" + str(fmin) + '-' + str(40) + '-nbSplit' + str(nbSplit) + ".csv"
             )

results_ImCoh_alone =  temp_results





#%% plot - TODO: put nodal???
results_theta_alpha_opt=results_theta_alpha_atm[(results_theta_alpha_atm["zthresh"]==float(df_res_opt_theta_alpha["zthresh"])) & (results_theta_alpha_atm["val_duration"]==float(df_res_opt_theta_alpha["val_duration"]))]
results_broad_opt=results_broad_atm[(results_broad_atm["zthresh"]==float(df_res_opt_db["zthresh"])) & (results_broad_atm["val_duration"]==float(df_res_opt_db["val_duration"]))]
results_theta_opt=results_theta_atm[(results_theta_atm["zthresh"]==float(df_res_opt_theta["zthresh"])) & (results_theta_atm["val_duration"]==float(df_res_opt_theta["val_duration"]))]
results_alpha_opt=results_alpha_atm[(results_alpha_atm["zthresh"]==float(df_res_opt_alpha["zthresh"])) & (results_alpha_atm["val_duration"]==float(df_res_opt_alpha["val_duration"]))]

results_opt_global = pd.concat((results_theta_alpha_opt, results_broad_opt, results_theta_opt, results_alpha_opt,
                    ))

res2plot = pd.concat((results_opt_global, results_ImCoh_alone))

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
                data=res2plot)

plt.savefig(path_figures_root + "Opt_HC_EP1_ATM_vs_ImCoh_SVM_Classification_2class-nbSplits"+str(nbSplit)+"_broad_MEG.pdf", dpi=300)

g = sns.catplot(y="test_precision",
                x='pipeline',
                hue="pipeline",
                kind='swarm',  # swarm
                col="freq",
                col_wrap=2,
                # dodge=True,
                #row='zthresh',
                height=4, aspect=3,
                data=res2plot)

plt.savefig(path_figures_root + "Opt_HC_EP1_ATM_vs_ImCoh_SVM_Precision_Classification_2class-nbSplits"+str(nbSplit)+"_broad_MEG.pdf", dpi=300)

g = sns.catplot(y="test_recall",
                x='pipeline',
                hue="pipeline",
                kind='swarm',  # swarm
                col="freq",
                col_wrap=2,
                # dodge=True,
                #row='zthresh',
                height=4, aspect=3,
                data=res2plot)

plt.savefig(path_figures_root + "Opt_HC_EP1_ATM_vs_ImCoh_SVM_Recall_Classification_2class-nbSplits"+str(nbSplit)+"_broad_MEG.pdf", dpi=300)

g = sns.catplot(y="test_roc_auc",
                x='pipeline',
                hue="pipeline",
                kind='swarm',  # swarm
                col="freq",
                col_wrap=2,
                # dodge=True,
                #row='zthresh',
                height=4, aspect=3,
                data=res2plot)

plt.savefig(path_figures_root + "Opt_HC_EP1_ATM_vs_ImCoh_SVM_ROC_AUC_Classification_2class-nbSplits"+str(nbSplit)+"_broad_MEG.pdf", dpi=300)


#%% infos to compare results
results_broad_opt["test_accuracy"].mean()
results_broad_opt["test_precision"].mean()
results_broad_opt["test_recall"].mean()
results_broad_opt["test_roc_auc"].mean()
results_broad_opt["test_f1"].mean()