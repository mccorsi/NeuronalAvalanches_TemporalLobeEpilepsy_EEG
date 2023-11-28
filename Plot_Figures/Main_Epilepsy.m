%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % MAIN - Plot results obtained from the analysis made
    % Authors: Marie-Constance Corsi
    % Date: 21/07/2023
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all; close all; clc;

%% paths to be updated
root_path='/Users/marieconstance.corsi/Documents/GitHub/NeuronalAvalanches_TemporalLobeEpilepsy_EEG/';
addpath(strcat(root_path,'Plot_Figures/Visualization_Cortex/'))

path_csv_root=strcat(root_path,'Database/');
fig_path=strcat(root_path,'Figures/');

cd(strcat(root_path,'/Plot_Figures'))

load(strcat(root_path,'/Scripts/Visualization_Cortex/cortex_15002V_MNI.mat'));
idx_DK=3;
nb_ROIs_DK=size(cortex_15002V_MNI.Atlas(idx_DK).Scouts,2);
labels_DK={cortex_15002V_MNI.Atlas(idx_DK).Scouts(1:nb_ROIs_DK).Label};

modality='EEG';
atlas='DK';
fs=250;

%% Fig 3.A. Plot feature importance (output from .py script)
color_ATM = [0.90796841, 0.49787195, 0.39171297]; % same colors as those chosen in Fig. 2
color_ImCoh = [0.29408557, 0.13721193, 0.38442775];

load(strcat(path_csv_root,'Features_ImCoh.mat'));
load(strcat(path_csv_root,'Features_ATM.mat')); % don't have the edges features for all the trial...

freq='broad band';
vmin=min(min(df_weights_estim_ImCoh_edges_nodal_broad_mean),min(df_weights_estim_ATM_edges_nodal_broad_mean));
vmax=max(max(df_weights_estim_ImCoh_edges_nodal_broad_mean),max(df_weights_estim_ATM_edges_nodal_broad_mean));
filename = strcat(path_figures_root,'Mean_Features_NodesFromEdges_HC_vs_EP1_Edges_ImCoh_freqband_',freq);
DoMyViz_node_Epilepsy(cortex_15002V_MNI, idx_DK, df_weights_estim_ImCoh_edges_nodal_broad_mean, filename, vmin, vmax)

filename = strcat(path_figures_root,'Mean_Features_NodesFromEdges_HC_vs_EP1_Edges_ATM_freqband_',freq);
DoMyViz_node_Epilepsy(cortex_15002V_MNI, idx_DK, df_weights_estim_ATM_edges_nodal_broad_mean, filename, vmin, vmax)


figure()
filename = strcat(path_figures_root,'Histogram_Features_HC_vs_EP1_Edges_ImCoh_freqband_', freq);
    histogram(df_weights_estim_ATM_edges_broad, 'Normalization', 'probability', 'FaceColor', color_ATM); hold on
    histogram(df_weights_estim_ImCoh_edges_broad, 'Normalization', 'probability', 'FaceColor', color_ImCoh);
    ylabel('probability');
    xlabel('feature importance value');
    legend('ATM','ImCoh');
    set(gca, 'YScale','log');
    box('off')
    legend boxoff
saveas(gcf,strcat(filename,'.pdf'));

%% Supplementary materials - narrow band
close all;
freq='theta-alpha band';

% mean case - paper
vmin=min(min(df_weights_estim_ImCoh_edges_nodal_theta_alpha_mean),min(df_weights_estim_ATM_edges_nodal_theta_alpha_mean));
vmax=max(max(df_weights_estim_ImCoh_edges_nodal_theta_alpha_mean),max(df_weights_estim_ATM_edges_nodal_theta_alpha_mean));
filename = strcat(path_figures_root,'Mean_Features_NodesFromEdges_HC_vs_EP1_Edges_ImCoh_freqband_',freq);
DoMyViz_node_Epilepsy(cortex_15002V_MNI, idx_DK, df_weights_estim_ImCoh_edges_nodal_broad_mean, filename, vmin, vmax)

filename = strcat(path_figures_root,'Mean_Features_NodesFromEdges_HC_vs_EP1_Edges_ATM_freqband_',freq);
DoMyViz_node_Epilepsy(cortex_15002V_MNI, idx_DK, df_weights_estim_ATM_edges_nodal_broad_mean, filename, vmin, vmax)


figure()
filename = strcat(path_figures_root,'Histogram_Features_HC_vs_EP1_Edges_ImCoh_freqband_', freq);
    histogram(df_weights_estim_ATM_edges_theta_alpha, 'Normalization', 'probability', 'FaceColor', color_ATM); hold on
    histogram(df_weights_estim_ImCoh_edges_theta_alpha, 'Normalization', 'probability', 'FaceColor', color_ImCoh);
    ylabel('probability');
    xlabel('feature importance value');
    legend('ATM','ImCoh');
    set(gca, 'YScale','log');
    box('off')
    legend boxoff
saveas(gcf,strcat(filename,'.pdf'));
