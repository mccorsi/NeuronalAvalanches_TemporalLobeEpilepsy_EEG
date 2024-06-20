"""
==============================================================
Plot figures - updated version post-rebuttal
===============================================================

"""
# Authors: Marie-Constance Corsi <marie-constance.corsi@inria.fr>
#
# License: BSD (3-clause)


import pandas as pd

import ptitprince as pt

import matplotlib.pyplot as plt
import scipy.io
import mne
import numpy as np
import mat73
from mne_connectivity.viz.circle import _plot_connectivity_circle
from mne.viz import circular_layout  # , plot_connectivity_circle

import os.path as osp
import os

import seaborn as sns


# %%
## functions
def isSquare(m): return all(len(row) == len(m) for row in m)


def regions_bst_to_lobes(argument):
    '''
        gather brainstorm labels in 5 lobes/areas
    '''
    switcher = {
        # gathers Left and Right in one single lobe
        # Frontal: includes both Frontal and PreFrontal in this way
        'LF': 'F',
        'RF': 'F',
        'LPF': 'F',
        'RPF': 'F',
        # Central: C --> Motor: M, except postcentral that is going to be placed in parietal...
        'LC': 'M',
        'RC': 'M',
        'M': 'M',  # in case that already done
        # Temporal: includes both Temporal and Lateral
        'LT': 'T',
        'RT': 'T',
        'LL': 'T',
        'RL': 'T',
        # Parietal
        'LP': 'P',
        'RP': 'P',
        'P': 'P',  # in case that already done
        # Occipital
        'LO': 'O',
        'RO': 'O'
    }
    return switcher.get(argument)


def Lobes_Partition(MatrixToBeDivided, idx_F, idx_M, idx_P, idx_T, idx_O):
    SubMatrix_F_F = MatrixToBeDivided[np.ix_(idx_F, idx_F)]
    SubMatrix_F_M = MatrixToBeDivided[np.ix_(idx_F, idx_M)]
    SubMatrix_F_T = MatrixToBeDivided[np.ix_(idx_F, idx_T)]
    SubMatrix_F_P = MatrixToBeDivided[np.ix_(idx_F, idx_P)]
    SubMatrix_F_O = MatrixToBeDivided[np.ix_(idx_F, idx_O)]

    # SubMatrix_M_F=MatrixToBeDivided[np.ix_(idx_M,idx_F)]
    SubMatrix_M_M = MatrixToBeDivided[np.ix_(idx_M, idx_M)]
    SubMatrix_M_T = MatrixToBeDivided[np.ix_(idx_M, idx_T)]
    SubMatrix_M_P = MatrixToBeDivided[np.ix_(idx_M, idx_P)]
    SubMatrix_M_O = MatrixToBeDivided[np.ix_(idx_M, idx_O)]

    # SubMatrix_T_F=MatrixToBeDivided[np.ix_(idx_T,idx_F)]
    # SubMatrix_T_M=MatrixToBeDivided[np.ix_(idx_T,idx_M)]
    SubMatrix_T_T = MatrixToBeDivided[np.ix_(idx_T, idx_T)]
    SubMatrix_T_P = MatrixToBeDivided[np.ix_(idx_T, idx_P)]
    SubMatrix_T_O = MatrixToBeDivided[np.ix_(idx_T, idx_O)]

    # SubMatrix_P_F=MatrixToBeDivided[np.ix_(idx_P,idx_F)]
    # SubMatrix_P_M=MatrixToBeDivided[np.ix_(idx_P,idx_M)]
    # SubMatrix_P_T=MatrixToBeDivided[np.ix_(idx_P,idx_T)]
    SubMatrix_P_P = MatrixToBeDivided[np.ix_(idx_P, idx_P)]
    SubMatrix_P_O = MatrixToBeDivided[np.ix_(idx_P, idx_O)]

    # SubMatrix_O_F=MatrixToBeDivided[np.ix_(idx_O,idx_F)]
    # SubMatrix_O_M=MatrixToBeDivided[np.ix_(idx_O,idx_M)]
    # SubMatrix_O_T=MatrixToBeDivided[np.ix_(idx_O,idx_T)]
    # SubMatrix_O_P=MatrixToBeDivided[np.ix_(idx_O,idx_P)]
    SubMatrix_O_O = MatrixToBeDivided[np.ix_(idx_O, idx_O)]

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

    return SubMatrix_F_F, SubMatrix_F_M, SubMatrix_F_T, SubMatrix_F_P, SubMatrix_F_O, \
        SubMatrix_M_M, SubMatrix_M_T, SubMatrix_M_P, SubMatrix_M_O, \
        SubMatrix_T_T, SubMatrix_T_P, SubMatrix_T_O, \
        SubMatrix_P_P, SubMatrix_P_O, \
        SubMatrix_O_O, output


# %% Set paths
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


# %% Set parameters linked to Desikan-Kiliany atlas
ROI_DK_list = ['bankssts L', 'bankssts R', 'caudalanteriorcingulate L', 'caudalanteriorcingulate R',
               'caudalmiddlefrontal L', 'caudalmiddlefrontal R', 'cuneus L', 'cuneus R', 'entorhinal L',
               'entorhinal R', 'frontalpole L', 'frontalpole R', 'fusiform L', 'fusiform R',
               'inferiorparietal L', 'inferiorparietal R', 'inferiortemporal L', 'inferiortemporal R', 'insula L',
               'insula R', 'isthmuscingulate L', 'isthmuscingulate R', 'lateraloccipital L', 'lateraloccipital R',
               'lateralorbitofrontal L', 'lateralorbitofrontal R', 'lingual L', 'lingual R',
               'medialorbitofrontal L', 'medialorbitofrontal R', 'middletemporal L', 'middletemporal R',
               'paracentral L', 'paracentral R', 'parahippocampal L', 'parahippocampal R', 'parsopercularis L',
               'parsopercularis R', 'parsorbitalis L', 'parsorbitalis R', 'parstriangularis L',
               'parstriangularis R', 'pericalcarine L', 'pericalcarine R', 'postcentral L', 'postcentral R',
               'posteriorcingulate L', 'posteriorcingulate R', 'precentral L', 'precentral R', 'precuneus L',
               'precuneus R', 'rostralanteriorcingulate L', 'rostralanteriorcingulate R', 'rostralmiddlefrontal L',
               'rostralmiddlefrontal R', 'superiorfrontal L', 'superiorfrontal R', 'superiorparietal L',
               'superiorparietal R', 'superiortemporal L', 'superiortemporal R', 'supramarginal L',
               'supramarginal R', 'temporalpole L', 'temporalpole R', 'transversetemporal L',
               'transversetemporal R']
ROIs = '%0d'.join(ROI_DK_list)
Region_DK = ['LT', 'RT', 'LL', 'RL', 'LF', 'RF', 'LO', 'RO', 'LT', 'RT', 'LPF', 'RPF', 'LT', 'RT', 'LP',
             'RP', 'LT', 'RT', 'LT', 'RT', 'LL', 'RL', 'LO', 'RO', 'LPF', 'RPF', 'LO', 'RO', 'LPF',
             'RPF', 'LT', 'RT', 'LC', 'RC', 'LT', 'RT', 'LF', 'RF', 'LPF', 'RPF', 'LF', 'RF', 'LO', 'RO', 'LC',
             'RC', 'LL', 'RL', 'LC', 'RC', 'LP', 'RP', 'LL', 'RL', 'LF', 'RF', 'LF', 'RF', 'LP', 'RP',
             'LT', 'RT', 'LP', 'RP', 'LT', 'RT', 'LT', 'RT']  # partition provided by Brainstorm

idx_DK_postcent = [i for i, j in enumerate(ROI_DK_list) if j == 'postcentral L' or j == 'postcentral R']
idx_DK_Maudalmiddlefront = [i for i, j in enumerate(ROI_DK_list) if
                            j == 'caudalmiddlefrontal L' or j == 'caudalmiddlefrontal R']
for i in idx_DK_postcent:
    Region_DK[i] = 'P'
for i in idx_DK_Maudalmiddlefront:
    Region_DK[i] = 'M'

# partition into lobes - DK
nb_ROIs_DK = len(Region_DK)
Lobes_DK = ["" for x in range(len(Region_DK))]
for kk_roi in range(len(Region_DK)):
    Lobes_DK[kk_roi] = regions_bst_to_lobes(Region_DK[kk_roi])

idx_F_DK = [i for i, j in enumerate(Lobes_DK) if j == 'F']
idx_M_DK = [i for i, j in enumerate(Lobes_DK) if j == 'M']
idx_P_DK = [i for i, j in enumerate(Lobes_DK) if j == 'P']
idx_O_DK = [i for i, j in enumerate(Lobes_DK) if j == 'O']
idx_T_DK = [i for i, j in enumerate(Lobes_DK) if j == 'T']
idx_lobes_DK = [idx_F_DK, idx_M_DK, idx_T_DK, idx_P_DK, idx_O_DK]

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
# %% Classification parameters
nbSplit = 50

# %% Figure 2A - Classification performance (accuracy) - ImCoh vs ATM
results = pd.DataFrame()
freqbands_tot = {'theta-alpha': [3, 14],
                 'paper': [3, 40]}

for f in freqbands_tot:
    fmin = freqbands_tot[f][0]
    fmax = freqbands_tot[f][1]
    temp_results = pd.read_csv(
        path_csv_root + "/SVM/Sensivity_Specificity_HC_EP1_IndivOpt_ATM_ImCoh_Comparison_SVM_Classification-allnode-2class-" +
        "-freq-" + str(fmin) + '-' + str(fmax) + '-nbSplit' + str(nbSplit) + ".csv"
        )

    results = pd.concat((results, temp_results))

pipelines = ["ATM+SVM", "ImCoh+SVM"]
freqs = ['3-14', '3-40']
#freqs = ['8-30', '3-30', '14-40']
results2plot = results[(results["pipeline"].isin(pipelines)) & (results["freq"].isin(freqs))]

dx = results2plot["test_accuracy_from_cm"]
dy = results2plot["pipeline"]
df = results2plot
ort = "h"
sigma = .2
cut = 0.
width = .6
orient = "h"

g = sns.FacetGrid(df, col="freq", height=5)
g = g.map_dataframe(pt.RainCloud, x=dy, y=dx, data=df, bw=sigma,
                    orient="h", palette='flare')
g.fig.subplots_adjust(top=0.75)
g.savefig(path_figures_root + 'AccuracyResults_RaincloudPlots.png', dpi=600, facecolor='white')
g.savefig(path_figures_root + 'Fig1A.eps', format='eps', dpi=600, facecolor='white')

# %% Figure 2B - ROC Curves - cf script 'Fig1B_Plot_ROC_Curves.py''


# %% Table 1
freqbands_tot = {'theta-alpha': [3, 14],
                 'alpha-beta': [8, 30],
                 'theta-alpha-beta': [3, 30],
                 'beta-gamma': [14, 40],
                 'paper': [3, 40]}
pipelines = ["ATM+SVM", "ImCoh+SVM"]
scoring = ["accuracy", "test_accuracy_from_cm", "precision", "recall", "f1", "roc_auc", "test_sensitivity",
           "test_specificity"]  # metrics used

results = pd.DataFrame()

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
    path_csv_root + "/SVM/OptConfig_HC_EP1_MEG_ATM_SVM_ClassificationRebuttal-allnode_rest_theta_alpha_Band.csv"
)
opt_atm_param_edge["theta-alpha"] = [df_res_opt_theta_alpha["zthresh"][0], df_res_opt_theta_alpha["val_duration"][0]]
perf_opt_atm_param_edge["theta-alpha"] = df_res_opt_theta_alpha["test_accuracy"][0]

df_res_opt_alpha_beta=pd.read_csv(
    path_csv_root + "/SVM/OptConfig_HC_EP1_MEG_ATM_SVM_ClassificationRebuttal-allnode_rest_alpha_beta_Band.csv"
)
opt_atm_param_edge["alpha-beta"] = [df_res_opt_alpha_beta["zthresh"][0], df_res_opt_alpha_beta["val_duration"][0]]
perf_opt_atm_param_edge["alpha-beta"] = df_res_opt_alpha_beta["test_accuracy"][0]

df_res_opt_beta_gamma=pd.read_csv(
    path_csv_root + "/SVM/OptConfig_HC_EP1_MEG_ATM_SVM_ClassificationRebuttal-allnode_rest_beta_gamma_Band.csv"
)
opt_atm_param_edge["beta-gamma"] = [df_res_opt_beta_gamma["zthresh"][0], df_res_opt_beta_gamma["val_duration"][0]]
perf_opt_atm_param_edge["beta-gamma"] = df_res_opt_beta_gamma["test_accuracy"][0]

df_res_opt_theta_alpha_beta=pd.read_csv(
    path_csv_root + "/SVM/OptConfig_HC_EP1_MEG_ATM_SVM_ClassificationRebuttal-allnode_rest_df_res_opt_theta_alpha_beta_Band.csv"
)
opt_atm_param_edge["theta-alpha-beta"] = [df_res_opt_theta_alpha_beta["zthresh"][0], df_res_opt_theta_alpha_beta["val_duration"][0]]
perf_opt_atm_param_edge["theta-alpha-beta"] = df_res_opt_theta_alpha_beta["test_accuracy"][0]

for f in freqbands_tot:
    fmin = freqbands_tot[f][0]
    fmax = freqbands_tot[f][1]

    k_zthresh_edge = opt_atm_param_edge[f][0]
    k_val_duration_edge = opt_atm_param_edge[f][1]

    temp_results = pd.read_csv(
        path_csv_root + "/SVM/HC_EP1_IndivOpt_ATM_ImCoh_Comparison_SVM_Classification-allnode-2class-" +
        "-freq-" + str(fmin) + '-' + str(fmax) + '-nbSplit' + str(nbSplit) + ".csv"
        )
    temp_results_atm = temp_results[temp_results["pipeline"] == "ATM+SVM"]
    temp_opt_atm = temp_results_atm[
        (temp_results_atm["zthresh"] == k_zthresh_edge) & (temp_results_atm["val_duration"] == k_val_duration_edge)]
    temp_results_imcoh = temp_results[temp_results["pipeline"] == "ImCoh+SVM"]

    results = pd.concat((results, temp_opt_atm, temp_results_imcoh))

    temp_results_ = pd.read_csv(
        path_csv_root + "/SVM/Sensivity_Specificity_HC_EP1_IndivOpt_ATM_ImCoh_Comparison_SVM_Classification-allnode-2class-" +
        "-freq-" + str(fmin) + '-' + str(fmax) + '-nbSplit' + str(nbSplit) + ".csv"
    )
    temp_results_additionalMetrics = temp_results_[temp_results_["pipeline"].isin(pipelines)]
    results = pd.concat((results, temp_results_additionalMetrics))

Res_Data_Table_ImCoh = dict()
Res_Data_Table_ATM = dict()
for f in freqbands_tot:
    fmin = freqbands_tot[f][0]
    fmax = freqbands_tot[f][1]

    Data_Table = results[results["freq"] == str(fmin) + "-" + str(fmax)]
    Data_Table_ATM = Data_Table[Data_Table["pipeline"] == "ATM+SVM"]
    Data_Table_ImCoh = Data_Table[Data_Table["pipeline"] == "ImCoh+SVM"]

    Mean_Data_Table_ATM = pd.DataFrame([Data_Table_ATM["test_accuracy"].mean(),
                                        Data_Table_ATM["test_accuracy_from_cm"].mean(),
                                        Data_Table_ATM["test_precision"].mean(),
                                        Data_Table_ATM["test_recall"].mean(),
                                        Data_Table_ATM["test_f1"].mean(),
                                        Data_Table_ATM["test_roc_auc"].mean(),
                                        Data_Table_ATM["test_sensitivity"].mean(),
                                        Data_Table_ATM["test_specificity"].mean()], columns=["avg"], index=scoring)

    Std_Data_Table_ATM = pd.DataFrame([
        Data_Table_ATM["test_accuracy"].std(),
        Data_Table_ATM["test_accuracy_from_cm"].std(),
        Data_Table_ATM["test_precision"].std(),
        Data_Table_ATM["test_recall"].std(),
        Data_Table_ATM["test_f1"].std(),
        Data_Table_ATM["test_roc_auc"].std(),
        Data_Table_ATM["test_sensitivity"].std(),
        Data_Table_ATM["test_specificity"].std()], columns=["std"], index=scoring)

    temp_Res_Data_Table_ATM = pd.concat((Mean_Data_Table_ATM, Std_Data_Table_ATM), axis=1)
    Res_Data_Table_ATM[str(fmin) + "-" + str(fmax)] = temp_Res_Data_Table_ATM

    Mean_Data_Table_ImCoh = pd.DataFrame([Data_Table_ImCoh["test_accuracy"].mean(),
                                          Data_Table_ImCoh["test_accuracy_from_cm"].mean(),
                                          Data_Table_ImCoh["test_precision"].mean(),
                                          Data_Table_ImCoh["test_recall"].mean(),
                                          Data_Table_ImCoh["test_f1"].mean(),
                                          Data_Table_ImCoh["test_roc_auc"].mean(),
                                          Data_Table_ImCoh["test_sensitivity"].mean(),
                                          Data_Table_ImCoh["test_specificity"].mean()], columns=["avg"], index=scoring)

    Std_Data_Table_ImCoh = pd.DataFrame([
        Data_Table_ImCoh["test_accuracy"].std(),
        Data_Table_ImCoh["test_accuracy_from_cm"].std(),
        Data_Table_ImCoh["test_precision"].std(),
        Data_Table_ImCoh["test_recall"].std(),
        Data_Table_ImCoh["test_f1"].std(),
        Data_Table_ImCoh["test_roc_auc"].std(),
        Data_Table_ImCoh["test_sensitivity"].std(),
        Data_Table_ImCoh["test_specificity"].std()], columns=["std"], index=scoring)

    temp_Res_Data_Table_ImCoh = pd.concat((Mean_Data_Table_ImCoh, Std_Data_Table_ImCoh), axis=1)
    Res_Data_Table_ImCoh[str(fmin) + "-" + str(fmax)] = temp_Res_Data_Table_ImCoh

Res_Data_Table = dict()
Res_Data_Table["ATM"] = Res_Data_Table_ATM
Res_Data_Table["ImCoh"] = Res_Data_Table_ImCoh

# %% Figure 3A - Features importance - Histograms - ImCoh vs ATM - cf Matlab script entitled 'Main_Epilepsy.m'

# %% Figure 3B - Features importance - Edge-wise - ImCoh vs ATM
nbSplit = 50
freqbands = {'paper': [3, 40]}
labels = pd.read_csv(path_csv_root + "LabelsDesikanAtlas.csv")
df_weights_estim_ATM_edges = dict()
df_weights_estim_ImCoh_edges = dict()

index_values = labels.values
column_values = labels.values

for f in freqbands:
    fmin = freqbands[f][0]
    fmax = freqbands[f][1]

    temp4 = pd.read_csv(
        path_csv_root + "/SVM/Features and co/ATM_edges_weights_FeaturesInfos_HC_EP1_SVM_Classification" + "-freq-" + str(
            fmin) + '-' + str(fmax) + '-nbSplit' + str(nbSplit) + ".csv")
    weights_estim_ATM_edges = np.array(temp4["median"]).reshape(68, 68)
    df_weights_estim_ATM_edges[f] = pd.DataFrame(data=weights_estim_ATM_edges,
                                                 index=index_values,
                                                 columns=column_values)

    temp2 = pd.read_csv(path_csv_root + "/SVM/ImCoh_edges_weights_HC_EP1_SVM_Classification" + "-freq-" + str(
        fmin) + '-' + str(fmax) + '-nbSplit' + str(nbSplit) + ".csv")
    weights_estim_ImCoh_edges = np.array(temp2["median"]).reshape(68, 68)
    df_weights_estim_ImCoh_edges[f] = pd.DataFrame(data=weights_estim_ImCoh_edges,
                                                   index=index_values,
                                                   columns=column_values)

    # test to identify the highest values
    threshold_val = 0.85
    sel_weights_estim_ImCoh_edges = np.zeros((nb_ROIs_DK, nb_ROIs_DK))
    idx_ImCoh = np.where(weights_estim_ImCoh_edges > threshold_val * np.max(weights_estim_ImCoh_edges))
    sel_weights_estim_ImCoh_edges[idx_ImCoh] = weights_estim_ImCoh_edges[idx_ImCoh]
    arr2 = np.count_nonzero(sel_weights_estim_ImCoh_edges)
    print(str(arr2 / 2) + ' edges > ' + str(threshold_val * 100) + '% of the max')
    df_sel_weights_estim_ImCoh_edges = pd.DataFrame(data=sel_weights_estim_ImCoh_edges,
                                                    index=index_values,
                                                    columns=column_values)
    df_sel_weights_estim_ImCoh_edges.to_csv(path_csv_root + "/SVM/Features and co/ImCoh_ranking" + "-freq-" + str(
        fmin) + '-' + str(fmax) + '-nbSplit' + str(nbSplit) + ".csv")

    threshold_val = 0.3
    sel_weights_estim_ATM_edges = np.zeros((nb_ROIs_DK, nb_ROIs_DK))
    idx_ATM = np.where(weights_estim_ATM_edges > threshold_val * np.max(weights_estim_ATM_edges))
    sel_weights_estim_ATM_edges[idx_ATM] = weights_estim_ATM_edges[idx_ATM]
    arr2 = np.count_nonzero(sel_weights_estim_ATM_edges)
    print(str(arr2 / 2) + ' edges > ' + str(threshold_val * 100) + '% of the max')
    df_sel_weights_estim_ATM_edges = pd.DataFrame(data=sel_weights_estim_ATM_edges,
                                                  index=index_values,
                                                  columns=column_values)
    df_sel_weights_estim_ATM_edges.to_csv(path_csv_root + "/SVM/Features and co/ATM_ranking" + "-freq-" + str(
        fmin) + '-' + str(fmax) + '-nbSplit' + str(nbSplit) + ".csv")

    # to ensure the same scale for both connectomes
    val_min = min(df_weights_estim_ATM_edges[f].T.values.min(), df_weights_estim_ImCoh_edges[f].T.values.min())
    val_max = max(df_weights_estim_ATM_edges[f].T.values.max(), df_weights_estim_ImCoh_edges[f].T.values.max())

    # plot connectome - ImCoh
    [SubMatrix_F_F, SubMatrix_F_M, SubMatrix_F_T, SubMatrix_F_P, SubMatrix_F_O, \
     SubMatrix_M_M, SubMatrix_M_T, SubMatrix_M_P, SubMatrix_M_O, \
     SubMatrix_T_T, SubMatrix_T_P, SubMatrix_T_O, \
     SubMatrix_P_P, SubMatrix_P_O, \
     SubMatrix_O_O, weights_estim_ImCoh_edges_2plot] = Lobes_Partition(weights_estim_ImCoh_edges, idx_F_DK, idx_M_DK,
                                                                       idx_P_DK, idx_T_DK, idx_O_DK)

    fig_edges_ImCoh, ax = plt.subplots(facecolor='white',
                                       subplot_kw=dict(polar=True))
    _plot_connectivity_circle(
        con=weights_estim_ImCoh_edges_2plot, node_names=node_names, node_colors=Region_DK_color_by_lobes,
        colorbar=False,  # True
        fontsize_names=7, fontsize_title=9,
        node_angles=node_angles,
        colormap='Greys', facecolor='white', textcolor='black',
        node_edgecolor='white', vmax=0.5 * val_max, vmin=val_min,
        fig=fig_edges_ImCoh)
    fig_edges_ImCoh.savefig(
        path_figures_root + 'Connect_Features_HC_vs_EP1_Edges_ImCoh_fmin_' + str(fmin) + '_fmax_' + str(fmax) + '.png',
        dpi=600,
        facecolor='white')
    fig_edges_ImCoh.savefig(
        path_figures_root + 'Fig2B_ImCoh_fmin_' + str(fmin) + '_fmax_' + str(fmax) + '.eps',
        format="eps",
        dpi=600,
        facecolor='white')

    # plot connectome - ATM
    [SubMatrix_F_F, SubMatrix_F_M, SubMatrix_F_T, SubMatrix_F_P, SubMatrix_F_O, \
     SubMatrix_M_M, SubMatrix_M_T, SubMatrix_M_P, SubMatrix_M_O, \
     SubMatrix_T_T, SubMatrix_T_P, SubMatrix_T_O, \
     SubMatrix_P_P, SubMatrix_P_O, \
     SubMatrix_O_O, weights_estim_ATM_edges_2plot] = Lobes_Partition(weights_estim_ATM_edges, idx_F_DK, idx_M_DK,
                                                                     idx_P_DK, idx_T_DK, idx_O_DK)

    fig_edges_ATM, ax = plt.subplots(facecolor='white',
                                     subplot_kw=dict(polar=True))
    _plot_connectivity_circle(
        con=weights_estim_ATM_edges_2plot, node_names=node_names, node_colors=Region_DK_color_by_lobes,
        colorbar=False,  # True
        fontsize_names=7, fontsize_title=9,
        node_angles=node_angles,
        colormap='Greys', facecolor='white', textcolor='black',
        node_edgecolor='white', vmax=0.5 * val_max, vmin=val_min,
        fig=fig_edges_ATM)
    fig_edges_ATM.savefig(
        path_figures_root + 'Connect_Features_HC_vs_EP1_Edges_ATM_fmin_' + str(fmin) + '_fmax_' + str(fmax) + '.png',
        dpi=600,
        facecolor='white')
    fig_edges_ATM.savefig(
        path_figures_root + 'Fig2B_ATM_fmin_' + str(fmin) + '_fmax_' + str(fmax) + '.eps',
        format="eps",
        dpi=600,
        facecolor='white')

# %% Figure 4 - Temporal scale - ATMs (ImCoh in suppl)
opt_trial_duration = [5, 15, 30, 60, 120, 180, 300]  # from 15 for ImCoh
freqbands = {'theta-alpha': [3, 14]}  # ,
# 'paper': [3, 40]}

results_trial_dur = pd.read_csv(
    path_csv_root + "/SVM/TrialDurationEffect_HC_EP1_IndivOpt_ATM__SVM_Classification-allnode-2class-" +
    '-nbSplit' + str(nbSplit) + ".csv"
    )
results_atm_edges = results_trial_dur[results_trial_dur["pipeline"] == "ATM+SVM"]
results_atm_nodal = results_trial_dur[results_trial_dur["pipeline"] == "ATM+SVM-nodal"]

# study by split
lst = list()
lst_nodal = list()
for f in freqbands:
    fmin = freqbands[f][0]
    fmax = freqbands[f][1]
    results_atm_edges_f = results_atm_edges[results_atm_edges["freq"] == str(fmin) + '-' + str(fmax)]

    # for a given duration: for each split, median over all the possible combinations of trial
    for kk_trial_crop in opt_trial_duration:
        temp_results_atm_edges_trial_dur = results_atm_edges_f[results_atm_edges_f["trial-duration"] == kk_trial_crop]

        for kk_split in range(nbSplit):
            temp_res_atm_edges = temp_results_atm_edges_trial_dur[temp_results_atm_edges_trial_dur["split"] == kk_split]
            score_edges = temp_res_atm_edges["test_accuracy"].median()
            pipeline_edges = temp_res_atm_edges["pipeline"].unique()[0]
            zthresh_edges = temp_res_atm_edges["zthresh"].unique()[0]
            val_duration_edges = temp_res_atm_edges["val_duration"].unique()[0]
            trial_duration_edges = temp_res_atm_edges["trial-duration"].unique()[0]
            freq_edges = temp_res_atm_edges["freq"].unique()[0]
            lst.append([score_edges, pipeline_edges, kk_split, zthresh_edges, val_duration_edges, freq_edges,
                        trial_duration_edges])

cols = ["test_accuracy", "pipeline", "split", "zthresh", "aval_duration", "freq", "trial_duration"]
pd_nb_bits_ATM_bySplit_edges = pd.DataFrame(lst, columns=cols)

# study by perm for a given definition of the trial
lst = list()
lst_nodal = list()
lst_imcoh = list()
lst_nodal_imcoh = list()
num_perm = 100
plt.close('all')
for f in freqbands:
    fmin = freqbands[f][0]
    fmax = freqbands[f][1]
    results_atm_edges_f = results_atm_edges[results_atm_edges["freq"] == str(fmin) + '-' + str(fmax)]

    # for a given duration: for each split, median over all the possible combinations of trial
    for kk_trial_crop in opt_trial_duration:
        temp_results_atm_edges_trial_dur = results_atm_edges_f[results_atm_edges_f["trial-duration"] == kk_trial_crop]

        for kk_trial_perm in range(num_perm):
            temp_res_atm_edges = temp_results_atm_edges_trial_dur[
                temp_results_atm_edges_trial_dur["trial-permutation"] == kk_trial_perm]
            score_edges = temp_res_atm_edges["test_accuracy"].median()
            pipeline_edges = temp_res_atm_edges["pipeline"].unique()[0]
            zthresh_edges = temp_res_atm_edges["zthresh"].unique()[0]
            val_duration_edges = temp_res_atm_edges["val_duration"].unique()[0]
            trial_duration_edges = temp_res_atm_edges["trial-duration"].unique()[0]
            freq_edges = temp_res_atm_edges["freq"].unique()[0]
            lst.append([score_edges, pipeline_edges, kk_trial_perm, zthresh_edges, val_duration_edges, freq_edges,
                        trial_duration_edges])

cols = ["test_accuracy", "pipeline", "trial-permutation", "zthresh", "aval_duration", "freq", "trial_duration"]
cols_imcoh = ["test_accuracy", "pipeline", "trial-permutation", "freq", "trial_duration"]
pd_nb_bits_ATM_byTrialPerm_edges = pd.DataFrame(lst, columns=cols)

# plot results
list_freq = pd_nb_bits_ATM_bySplit_edges["freq"].unique()
plt.style.use("classic")
for frequency in list_freq:
    temp_pd_nb_bits_ATM_bySplit_edges = pd_nb_bits_ATM_bySplit_edges[pd_nb_bits_ATM_bySplit_edges["freq"] == frequency]
    g = sns.catplot(y="test_accuracy",
                    x='trial_duration',
                    height=4, aspect=3, s=2,
                    palette = 'colorblind',
                    data=temp_pd_nb_bits_ATM_bySplit_edges)
    plt.savefig(path_figures_root + "Paper_TrialDurationEffect_BySplit_HC_EP1_ATM_SVM_Classification_edges-2class-nbSplits"+str(nbSplit)+"_"+frequency+".pdf", dpi=300)

    sns.set_style('white')
    palette = 'Blues'
    ax = sns.violinplot(y="test_accuracy", x="trial_duration", data=temp_pd_nb_bits_ATM_bySplit_edges,
                        hue="trial_duration", dodge=False,
                        palette=palette,
                        scale="width", inner=None)
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    for violin in ax.collections:
        bbox = violin.get_paths()[0].get_extents()
        x0, y0, width, height = bbox.bounds
        violin.set_clip_path(plt.Rectangle((x0, y0), width / 2, height, transform=ax.transData))

    # sns.boxplot(y="test_accuracy", x="trial_duration", data=temp_pd_nb_bits_ATM_bySplit_edges, saturation=1, showfliers=False,
    #             width=0.3, boxprops={'zorder': 3, 'facecolor': 'none'}, ax=ax)
    old_len_collections = len(ax.collections)
    sns.stripplot(y="test_accuracy", x="trial_duration", data=temp_pd_nb_bits_ATM_bySplit_edges,
                  hue="trial_duration", palette=palette, dodge=False, ax=ax, size=3)
    for dots in ax.collections[old_len_collections:]:
        dots.set_offsets(dots.get_offsets() + np.array([0.18, 0]))
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.legend_.remove()
    plt.savefig(
        path_figures_root + "Prova1_Paper_TrialDurationEffect_BySplit_HC_EP1_ATM_SVM_Classification_edges-2class-nbSplits" + str(
            nbSplit) + "_" + frequency + ".pdf", dpi=300)
    plt.savefig(
        path_figures_root + "Prova1_Paper_TrialDurationEffect_BySplit_HC_EP1_ATM_SVM_Classification_edges-2class-nbSplits" + str(
            nbSplit) + "_" + frequency + ".png", dpi=300)
    plt.savefig(path_figures_root + "Fig3_TrialDurationEffect_BySplit_" + frequency + ".eps", format="eps", dpi=300)
    plt.close('all')

    temp_pd_nb_bits_ATM_byTrialPerm_edges = pd_nb_bits_ATM_byTrialPerm_edges[
        pd_nb_bits_ATM_byTrialPerm_edges["freq"] == frequency]
    g = sns.catplot(y="test_accuracy",
                    x='trial_duration',
                    height=4, aspect=3, s=2,
                    palette = 'colorblind',
                    data=pd_nb_bits_ATM_byTrialPerm_edges)
    plt.savefig(path_figures_root + "Paper_TrialDurationEffect_ByTrialPerm_HC_EP1_ATM_SVM_Classification_edges-2class-nbSplits"+str(nbSplit)+"_"+frequency+".pdf", dpi=300)

# %% Supplementary - Average ATM & ImCoh
grp_id_2use = ['HC', 'EPI 1']
precomp_concat_name = path_data_root + '/concat_epochs_HC_EP1.gz'

if osp.exists(precomp_concat_name):
    print("Loading existing concatenated precomputations...")
    with gzip.open(precomp_concat_name, "rb") as file:
        epochs_concat, labels_concat = pickle.load(file)
else:
    print("Please consider performing precomputations and/or concatenation of the epochs!")

labels = pd.read_csv(path_csv_root + "LabelsDesikanAtlas.csv")
index_values = labels.values
column_values = labels.values

df_Avg_ATM_HC_edges = dict()
df_Avg_ATM_EP1_edges = dict()
df_Avg_ATM_HC_edges_nodal = dict()
df_Avg_ATM_EP1_edges_nodal = dict()

for kk_trial_crop in opt_trial_duration:
    tmin = 0
    tmax = kk_trial_crop

    for f in freqbands:
        fmin = freqbands[f][0]
        fmax = freqbands[f][1]
        epochs2use = epochs_concat[f].drop([2, 31, 32, 33, 34],
                                           reason='USER')  # to have the same nb of pz/subjects and because data from the second patient may be corrupted
        epochs_crop = epochs2use.crop(tmin=tmin, tmax=tmax, include_tmax=True)

        epochs_in_permuted_order = epochs_crop
        data = epochs_crop.get_data()
        label = labels_concat[f]
        nb_trials = len(data)  # nb subjects/ pz here
        nb_ROIs = np.shape(data)[1]

        ### ATM + SVM
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

        # ATM computation with edge-wise optimization parameters
        ATM = np.empty((nb_trials, nb_ROIs, nb_ROIs))
        for kk_trial in range(nb_trials):
            list_avalanches, min_size_avalanches, max_size_avalanches, list_avalanches_bysize, temp_ATM = find_avalanches(
                zscored_data_ep[kk_trial, :, :], thresh=kk_zthresh_edge, val_duration=kk_val_duration_edge)
            # ATM: nb_trials x nb_ROIs x nb_ROIs matrix
            ATM[kk_trial, :, :] = temp_ATM

        fixed_reshape_ATM = np.nan_to_num(ATM, nan=0)  # replace nan by 0

        ATM_HC = fixed_reshape_ATM[:31, :, :]
        ATM_EP1 = fixed_reshape_ATM[30:nb_trials, :, :]

        Avg_ATM_HC_edges = np.mean(ATM_HC, 0)
        Avg_ATM_HC_edges_nodal = np.sum(Avg_ATM_HC_edges, 0)
        Avg_ATM_EP1_edges = np.mean(ATM_EP1, 0)
        Avg_ATM_EP1_edges_nodal = np.sum(Avg_ATM_EP1_edges, 0)

        df_Avg_ATM_HC_edges[f] = pd.DataFrame(data=Avg_ATM_HC_edges,
                                              index=index_values,
                                              columns=column_values)

        df_Avg_ATM_HC_edges_nodal[f] = pd.DataFrame(data=Avg_ATM_HC_edges_nodal,
                                                    index=index_values,
                                                    columns=["ROIs"])

        df_Avg_ATM_EP1_edges[f] = pd.DataFrame(data=Avg_ATM_EP1_edges,
                                               index=index_values,
                                               columns=column_values)

        df_Avg_ATM_EP1_edges_nodal[f] = pd.DataFrame(data=Avg_ATM_EP1_edges_nodal,
                                                     index=index_values,
                                                     columns=["ROIs"])

        vmin = np.min((df_Avg_ATM_EP1_edges[f], df_Avg_ATM_HC_edges[f]))
        vmax = np.max((df_Avg_ATM_EP1_edges[f], df_Avg_ATM_HC_edges[f]))

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
            vmin=vmin, vmax=vmax,
            fig=fig_edges_HC)
        fig_edges_HC.savefig(
            path_figures_root + 'Connect_Avg_HC_Edges_ATM_fmin_' + str(fmin) + '_fmax_' + str(fmax) + '.png',
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
            vmin=vmin, vmax=vmax,
            fig=fig_edges_EP1)
        fig_edges_EP1.savefig(
            path_figures_root + 'Connect_Avg_EP1_Edges_ATM_fmin_' + str(fmin) + '_fmax_' + str(fmax) + '.png',
            dpi=600,
            facecolor='white')

# ImCoh

df_Avg_ImCoh_HC_edges = dict()
df_Avg_ImCoh_EP1_edges = dict()
df_Avg_ImCoh_HC_edges_nodal = dict()
df_Avg_ImCoh_EP1_edges_nodal = dict()

for kk_trial_crop in opt_trial_duration:
    tmin = 0
    tmax = kk_trial_crop

    for f in freqbands:
        fmin = freqbands[f][0]
        fmax = freqbands[f][1]
        epochs2use = epochs_concat[f].drop([2, 31, 32, 33, 34],
                                           reason='USER')  # to have the same nb of pz/subjects and because data from the second patient may be corrupted
        epochs_crop = epochs2use.crop(tmin=tmin, tmax=tmax, include_tmax=True)

        epochs_in_permuted_order = epochs_crop
        data = epochs_crop.get_data()
        label = labels_concat[f]
        nb_trials = len(data)  # nb subjects/ pz here
        nb_ROIs = np.shape(data)[1]

        ### ImCoh + SVM
        # # here to compute FC on long recordings we increase the time window of interest to speed up the process
        ft = FunctionalTransformer(
            delta=10, ratio=0.5, method="imcoh", fmin=fmin, fmax=fmax
        )
        preproc_meg = Pipeline(steps=[("ft", ft)])  # , ("spd", EnsureSPD())])
        mat_ImCoh = preproc_meg.fit_transform(epochs_in_permuted_order)

        ImCoh_cl = mat_ImCoh

        ImCoh_HC = ImCoh_cl[:31, :, :]
        ImCoh_EP1 = ImCoh_cl[30:nb_trials, :, :]

        Avg_ImCoh_HC_edges = np.mean(ImCoh_HC, 0)
        Avg_ImCoh_HC_edges_nodal = np.sum(Avg_ImCoh_HC_edges, 0)
        Avg_ImCoh_EP1_edges = np.mean(ImCoh_EP1, 0)
        Avg_ImCoh_EP1_edges_nodal = np.sum(Avg_ImCoh_EP1_edges, 0)

        df_Avg_ImCoh_HC_edges[f] = pd.DataFrame(data=Avg_ImCoh_HC_edges,
                                              index=index_values,
                                              columns=column_values)

        df_Avg_ImCoh_HC_edges_nodal[f] = pd.DataFrame(data=Avg_ImCoh_HC_edges_nodal,
                                                    index=index_values,
                                                    columns=["ROIs"])

        df_Avg_ImCoh_EP1_edges[f] = pd.DataFrame(data=Avg_ImCoh_EP1_edges,
                                               index=index_values,
                                               columns=column_values)

        df_Avg_ImCoh_EP1_edges_nodal[f] = pd.DataFrame(data=Avg_ImCoh_EP1_edges_nodal,
                                                     index=index_values,
                                                     columns=["ROIs"])

        # plot connectome
        [SubMatrix_F_F, SubMatrix_F_M, SubMatrix_F_T, SubMatrix_F_P, SubMatrix_F_O, \
         SubMatrix_M_M, SubMatrix_M_T, SubMatrix_M_P, SubMatrix_M_O, \
         SubMatrix_T_T, SubMatrix_T_P, SubMatrix_T_O, \
         SubMatrix_P_P, SubMatrix_P_O, \
         SubMatrix_O_O, Avg_ImCoh_HC_edges_2plot] = Lobes_Partition(Avg_ImCoh_HC_edges, idx_F_DK, idx_M_DK, idx_P_DK,
                                                                  idx_T_DK, idx_O_DK)

        np.fill_diagonal(Avg_ImCoh_HC_edges_2plot, 0)
        valmin=Avg_ImCoh_HC_edges_2plot.min()
        valmax=Avg_ImCoh_HC_edges_2plot.max()
        vmax = max(abs(valmin), abs(valmax))
        fig_edges_HC, ax = plt.subplots(facecolor='white',
                                        subplot_kw=dict(polar=True))
        _plot_connectivity_circle(
            con=Avg_ImCoh_HC_edges_2plot, node_names=node_names, node_colors=Region_DK_color_by_lobes,
            colorbar=True,
            fontsize_names=7, fontsize_title=9,
            node_angles=node_angles,
            colormap='RdBu_r', facecolor='white', textcolor='black',
            node_edgecolor='white',
            vmin=-vmax,
            vmax=vmax,
            fig=fig_edges_HC)
        fig_edges_HC.savefig(
            path_figures_root + 'Connect_Avg_HC_Edges_ImCoh_fmin_' + str(fmin) + '_fmax_' + str(fmax) + '.png',
            dpi=600,
            facecolor='white')

        # plot connectome
        [SubMatrix_F_F, SubMatrix_F_M, SubMatrix_F_T, SubMatrix_F_P, SubMatrix_F_O, \
         SubMatrix_M_M, SubMatrix_M_T, SubMatrix_M_P, SubMatrix_M_O, \
         SubMatrix_T_T, SubMatrix_T_P, SubMatrix_T_O, \
         SubMatrix_P_P, SubMatrix_P_O, \
         SubMatrix_O_O, Avg_ImCoh_EP1_edges_2plot] = Lobes_Partition(Avg_ImCoh_EP1_edges, idx_F_DK, idx_M_DK, idx_P_DK,
                                                                   idx_T_DK, idx_O_DK)

        np.fill_diagonal(Avg_ImCoh_EP1_edges_2plot, 0)
        valmin=Avg_ImCoh_EP1_edges_2plot.min()
        valmax=Avg_ImCoh_EP1_edges_2plot.max()
        vmax = max(abs(valmin), abs(valmax))
        fig_edges_EP1, ax = plt.subplots(facecolor='white',
                                         subplot_kw=dict(polar=True))
        _plot_connectivity_circle(
            con=Avg_ImCoh_EP1_edges_2plot, node_names=node_names, node_colors=Region_DK_color_by_lobes,
            colorbar=True,
            fontsize_names=7, fontsize_title=9,
            node_angles=node_angles,
            colormap='RdBu_r', facecolor='white', textcolor='black',
            node_edgecolor='white',
            vmin=-vmax,
            vmax=vmax,
            fig=fig_edges_EP1)
        fig_edges_EP1.savefig(
            path_figures_root + 'Connect_Avg_EP1_Edges_ImCoh_fmin_' + str(fmin) + '_fmax_' + str(fmax) + '.png',
            dpi=600,
            facecolor='white')
# %% Supplementary - ATM & ImCoh - Features study - Narrow band
nbSplit = 50
freqbands = {'theta-alpha': [3, 14]}
labels = pd.read_csv(path_csv_root + "LabelsDesikanAtlas.csv")
df_weights_estim_ATM_edges = dict()
df_weights_estim_ImCoh_edges = dict()

index_values = labels.values
column_values = labels.values

for f in freqbands:
    fmin = freqbands[f][0]
    fmax = freqbands[f][1]

    temp4 = pd.read_csv(
        path_csv_root + "/SVM/Features and co/ATM_edges_weights_FeaturesInfos_HC_EP1_SVM_Classification" + "-freq-" + str(
            fmin) + '-' + str(fmax) + '-nbSplit' + str(nbSplit) + ".csv")
    weights_estim_ATM_edges = np.array(temp4["median"]).reshape(68, 68)
    df_weights_estim_ATM_edges[f] = pd.DataFrame(data=weights_estim_ATM_edges,
                                                 index=index_values,
                                                 columns=column_values)

    temp2 = pd.read_csv(path_csv_root + "/SVM/ImCoh_edges_weights_HC_EP1_SVM_Classification" + "-freq-" + str(
        fmin) + '-' + str(fmax) + '-nbSplit' + str(nbSplit) + ".csv")
    weights_estim_ImCoh_edges = np.array(temp2["median"]).reshape(68, 68)
    df_weights_estim_ImCoh_edges[f] = pd.DataFrame(data=weights_estim_ImCoh_edges,
                                                   index=index_values,
                                                   columns=column_values)

    # test to identify the highest values
    threshold_val = 0.85
    sel_weights_estim_ImCoh_edges = np.zeros((nb_ROIs_DK, nb_ROIs_DK))
    idx_ImCoh = np.where(weights_estim_ImCoh_edges > threshold_val * np.max(weights_estim_ImCoh_edges))
    sel_weights_estim_ImCoh_edges[idx_ImCoh] = weights_estim_ImCoh_edges[idx_ImCoh]
    arr2 = np.count_nonzero(sel_weights_estim_ImCoh_edges)
    print(str(arr2 / 2) + ' edges > ' + str(threshold_val * 100) + '% of the max')
    df_sel_weights_estim_ImCoh_edges = pd.DataFrame(data=sel_weights_estim_ImCoh_edges,
                                                    index=index_values,
                                                    columns=column_values)
    df_sel_weights_estim_ImCoh_edges.to_csv(path_csv_root + "/SVM/Features and co/ImCoh_ranking" + "-freq-" + str(
        fmin) + '-' + str(fmax) + '-nbSplit' + str(nbSplit) + ".csv")

    threshold_val = 0.3
    sel_weights_estim_ATM_edges = np.zeros((nb_ROIs_DK, nb_ROIs_DK))
    idx_ATM = np.where(weights_estim_ATM_edges > threshold_val * np.max(weights_estim_ATM_edges))
    sel_weights_estim_ATM_edges[idx_ATM] = weights_estim_ATM_edges[idx_ATM]
    arr2 = np.count_nonzero(sel_weights_estim_ATM_edges)
    print(str(arr2 / 2) + ' edges > ' + str(threshold_val * 100) + '% of the max')
    df_sel_weights_estim_ATM_edges = pd.DataFrame(data=sel_weights_estim_ATM_edges,
                                                  index=index_values,
                                                  columns=column_values)
    df_sel_weights_estim_ATM_edges.to_csv(path_csv_root + "/SVM/Features and co/ATM_ranking" + "-freq-" + str(
        fmin) + '-' + str(fmax) + '-nbSplit' + str(nbSplit) + ".csv")

    # to ensure the same scale for both connectomes
    val_min = min(df_weights_estim_ATM_edges[f].T.values.min(), df_weights_estim_ImCoh_edges[f].T.values.min())
    val_max = max(df_weights_estim_ATM_edges[f].T.values.max(), df_weights_estim_ImCoh_edges[f].T.values.max())

    # plot connectome - ImCoh
    [SubMatrix_F_F, SubMatrix_F_M, SubMatrix_F_T, SubMatrix_F_P, SubMatrix_F_O, \
     SubMatrix_M_M, SubMatrix_M_T, SubMatrix_M_P, SubMatrix_M_O, \
     SubMatrix_T_T, SubMatrix_T_P, SubMatrix_T_O, \
     SubMatrix_P_P, SubMatrix_P_O, \
     SubMatrix_O_O, weights_estim_ImCoh_edges_2plot] = Lobes_Partition(weights_estim_ImCoh_edges, idx_F_DK, idx_M_DK,
                                                                       idx_P_DK, idx_T_DK, idx_O_DK)

    fig_edges_ImCoh, ax = plt.subplots(facecolor='white',
                                       subplot_kw=dict(polar=True))
    _plot_connectivity_circle(
        con=weights_estim_ImCoh_edges_2plot, node_names=node_names, node_colors=Region_DK_color_by_lobes,
        colorbar=False,  # True
        fontsize_names=7, fontsize_title=9,
        node_angles=node_angles,
        colormap='Greys', facecolor='white', textcolor='black',
        node_edgecolor='white', vmax=0.5 * val_max, vmin=val_min,
        fig=fig_edges_ImCoh)
    fig_edges_ImCoh.savefig(
        path_figures_root + 'Connect_Features_HC_vs_EP1_Edges_ImCoh_fmin_' + str(fmin) + '_fmax_' + str(fmax) + '.png',
        dpi=600,
        facecolor='white')
    fig_edges_ImCoh.savefig(
        path_figures_root + 'Suppl_ImCoh_fmin_' + str(fmin) + '_fmax_' + str(fmax) + '.eps',
        format="eps",
        dpi=600,
        facecolor='white')

    # plot connectome - ATM
    [SubMatrix_F_F, SubMatrix_F_M, SubMatrix_F_T, SubMatrix_F_P, SubMatrix_F_O, \
     SubMatrix_M_M, SubMatrix_M_T, SubMatrix_M_P, SubMatrix_M_O, \
     SubMatrix_T_T, SubMatrix_T_P, SubMatrix_T_O, \
     SubMatrix_P_P, SubMatrix_P_O, \
     SubMatrix_O_O, weights_estim_ATM_edges_2plot] = Lobes_Partition(weights_estim_ATM_edges, idx_F_DK, idx_M_DK,
                                                                     idx_P_DK, idx_T_DK, idx_O_DK)

    fig_edges_ATM, ax = plt.subplots(facecolor='white',
                                     subplot_kw=dict(polar=True))
    _plot_connectivity_circle(
        con=weights_estim_ATM_edges_2plot, node_names=node_names, node_colors=Region_DK_color_by_lobes,
        colorbar=False,  # True
        fontsize_names=7, fontsize_title=9,
        node_angles=node_angles,
        colormap='Greys', facecolor='white', textcolor='black',
        node_edgecolor='white', vmax=0.5 * val_max, vmin=val_min,
        fig=fig_edges_ATM)
    fig_edges_ATM.savefig(
        path_figures_root + 'Connect_Features_HC_vs_EP1_Edges_ATM_fmin_' + str(fmin) + '_fmax_' + str(fmax) + '.png',
        dpi=600,
        facecolor='white')
    fig_edges_ATM.savefig(
        path_figures_root + 'Suppl_ATM_fmin_' + str(fmin) + '_fmax_' + str(fmax) + '.eps',
        format="eps",
        dpi=600,
        facecolor='white')

# %% Supplementary - ImCoh - Temporal scale
opt_trial_duration = [15, 30, 60, 120, 180, 300]  # 5 not done

results_trial_dur = pd.read_csv(
    path_csv_root + "/SVM/TrialDurationEffect_HC_EP1_IndivOpt_ImCoh_Classification-allnode-2class-" +
    '-nbSplit' + str(nbSplit) + ".csv"
    )
results_imcoh_edges = results_trial_dur[results_trial_dur["pipeline"] == "ImCoh+SVM"]

# study by split
lst = list()
lst_nodal = list()
for f in freqbands:
    fmin = freqbands[f][0]
    fmax = freqbands[f][1]
    results_imcoh_edges_f = results_imcoh_edges[results_imcoh_edges["freq"] == str(fmin) + '-' + str(fmax)]
    # for a given duration: for each split, median over all the possible combinations of trial
    for kk_trial_crop in opt_trial_duration:
        temp_results_imcoh_edges_trial_dur = results_imcoh_edges_f[
            results_imcoh_edges_f["trial-duration"] == kk_trial_crop]
        for kk_split in range(nbSplit):
            temp_res_imcoh_edges = temp_results_imcoh_edges_trial_dur[
                temp_results_imcoh_edges_trial_dur["split"] == kk_split]
            score_edges = temp_res_imcoh_edges["test_accuracy"].median()
            trial_duration_edges = temp_res_imcoh_edges["trial-duration"].unique()[0]
            freq_edges = temp_res_imcoh_edges["freq"].unique()[0]
            lst.append([score_edges, kk_split, freq_edges, trial_duration_edges])

cols = ["test_accuracy", "split", "freq", "trial_duration"]
pd_nb_bits_ImCoh_bySplit_edges = pd.DataFrame(lst, columns=cols)

# study by perm for a given definition of the trial
lst = list()
lst_nodal = list()
num_perm = 100
for f in freqbands:
    fmin = freqbands[f][0]
    fmax = freqbands[f][1]
    results_imcoh_edges_f = results_imcoh_edges[results_imcoh_edges["freq"] == str(fmin) + '-' + str(fmax)]

    # for a given duration: for each split, median over all the possible combinations of trial
    for kk_trial_crop in opt_trial_duration:
        temp_results_imcoh_edges_trial_dur = results_imcoh_edges_f[
            results_imcoh_edges_f["trial-duration"] == kk_trial_crop]

        for kk_trial_perm in range(num_perm):
            temp_res_imcoh_edges = temp_results_imcoh_edges_trial_dur[
                temp_results_imcoh_edges_trial_dur["trial-permutation"] == kk_trial_perm]
            score_edges = temp_res_imcoh_edges["test_accuracy"].median()
            trial_duration_edges = temp_res_imcoh_edges["trial-duration"].unique()[0]
            freq_edges = temp_res_imcoh_edges["freq"].unique()[0]
            lst.append([score_edges, kk_trial_perm, freq_edges, trial_duration_edges])

cols = ["test_accuracy", "trial-permutation", "freq", "trial_duration"]
pd_nb_bits_ImCoh_byTrialPerm_edges = pd.DataFrame(lst, columns=cols)

# plot results
list_freq = pd_nb_bits_ImCoh_bySplit_edges["freq"].unique()
plt.style.use("classic")
for frequency in list_freq:
    temp_pd_nb_bits_ImCoh_bySplit_edges = pd_nb_bits_ImCoh_bySplit_edges[
        pd_nb_bits_ImCoh_bySplit_edges["freq"] == frequency]
    g = sns.catplot(y="test_accuracy",
                    x='trial_duration',
                    height=4, aspect=3, s=2,
                    palette='colorblind',
                    data=temp_pd_nb_bits_ImCoh_bySplit_edges)
    plt.savefig(
        path_figures_root + "SI_TrialDurationEffect_BySplit_HC_EP1_imcoh_SVM_Classification_edges-2class-nbSplits" + str(
            nbSplit) + "_" + frequency + ".pdf", dpi=300)

    temp_pd_nb_bits_ImCoh_byTrialPerm_edges = pd_nb_bits_ImCoh_byTrialPerm_edges[
        pd_nb_bits_ImCoh_byTrialPerm_edges["freq"] == frequency]
    g = sns.catplot(y="test_accuracy",
                    x='trial_duration',
                    height=4, aspect=3, s=2,
                    palette='colorblind',
                    data=pd_nb_bits_ImCoh_byTrialPerm_edges)
    plt.savefig(
        path_figures_root + "SI_TrialDurationEffect_ByTrialPerm_HC_EP1_imcoh_SVM_Classification_edges-2class-nbSplits" + str(
            nbSplit) + "_" + frequency + ".pdf", dpi=300)

# %%
nbSplit = 50
freqbands = {'theta': [3, 8],
             'alpha': [8, 14],
             'theta-alpha': [3, 14],  # Epilepsy case
             'paper': [3, 40]}

labels = pd.read_csv(path_csv_root + "LabelsDesikanAtlas.csv")
df_weights_estim_ImCoh_nodal = dict()
df_weights_estim_ImCoh_edges = dict()
df_weights_estim_ImCoh_edges_nodal = dict()
df_weights_estim_ATM_nodal = dict()
df_weights_estim_ATM_edges = dict()
df_weights_estim_ATM_edges_nodal = dict()
df_weights_estim_ATM_ImCoh_edges_nodal = dict()

index_values = labels.values
column_values = labels.values

for f in freqbands:
    fmin = freqbands[f][0]
    fmax = freqbands[f][1]

    temp = pd.read_csv(path_csv_root + "/SVM/ImCoh_nodal_weights_HC_EP1_SVM_Classification" + "-freq-" + str(
        fmin) + '-' + str(fmax) + '-nbSplit' + str(nbSplit) + ".csv")
    weights_estim_ImCoh_nodal = temp["median"]
    df_weights_estim_ImCoh_nodal[f] = pd.DataFrame(data=weights_estim_ImCoh_nodal)
    temp2 = pd.read_csv(path_csv_root + "/SVM/ImCoh_edges_weights_HC_EP1_SVM_Classification" + "-freq-" + str(
        fmin) + '-' + str(fmax) + '-nbSplit' + str(nbSplit) + ".csv")
    weights_estim_ImCoh_edges = np.array(temp2["median"]).reshape(68, 68)
    df_weights_estim_ImCoh_edges[f] = pd.DataFrame(data=weights_estim_ImCoh_edges,
                                                   index=index_values,
                                                   columns=column_values)

    weights_estim_ImCoh_edges_nodal = np.mean(weights_estim_ImCoh_edges, 1)
    df_weights_estim_ImCoh_edges_nodal[f] = pd.DataFrame(data=weights_estim_ImCoh_edges_nodal,
                                                         index=index_values,
                                                         columns=["ROIs-ImCoh"])

    temp3 = pd.read_csv(
        path_csv_root + "/SVM/Features and co/ATM_nodal_weights_FeaturesInfos_HC_EP1_SVM_Classification" + "-freq-" + str(
            fmin) + '-' + str(fmax) + '-nbSplit' + str(nbSplit) + ".csv")
    weights_estim_ATM_nodal = temp3["median"]
    df_weights_estim_ATM_nodal[f] = pd.DataFrame(data=weights_estim_ATM_nodal)

    temp4 = pd.read_csv(
        path_csv_root + "/SVM/Features and co/ATM_edges_weights_FeaturesInfos_HC_EP1_SVM_Classification" + "-freq-" + str(
            fmin) + '-' + str(fmax) + '-nbSplit' + str(nbSplit) + ".csv")
    weights_estim_ATM_edges = np.array(temp4["median"]).reshape(68, 68)
    df_weights_estim_ATM_edges[f] = pd.DataFrame(data=weights_estim_ATM_edges,
                                                 index=index_values,
                                                 columns=column_values)

    weights_estim_ATM_edges_nodal = np.mean(weights_estim_ATM_edges, 1)
    df_weights_estim_ATM_edges_nodal[f] = pd.DataFrame(data=weights_estim_ATM_edges_nodal,
                                                       index=index_values,
                                                       columns=["ROIs-ATM"])

    df_weights_estim_ImCoh_edges_nodal[f].reset_index(drop=True, inplace=True)
    df_weights_estim_ATM_edges_nodal[f].reset_index(drop=True, inplace=True)
    df_weights_estim_ATM_ImCoh_edges_nodal[f] = pd.concat(
        (df_weights_estim_ImCoh_edges_nodal[f], df_weights_estim_ATM_edges_nodal[f]), axis=1)
    df_weights_estim_ATM_ImCoh_edges_nodal[f].index = index_values

df_weights_estim_ATM_ImCoh_edges_nodal[f].to_csv(
    path_csv_root + 'Mean_df_weights_estim_ATM_ImCoh_edges_nodal_broad.csv')