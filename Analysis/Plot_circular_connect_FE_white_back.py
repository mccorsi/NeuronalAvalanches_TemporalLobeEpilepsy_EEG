# TODO: https://towardsdatascience.com/chord-diagrams-of-protein-interaction-networks-in-python-9589affc8b91
 # https://github.com/ericmjl/nxviz/blob/master/examples/circos/group_labels.py
# create a script from a matrix and a vector of labels - plot connectome

import pandas as pd
import matplotlib.pyplot as plt
import scipy.io
import mne
import numpy as np
#%%
root_path='/Users/marieconstance.corsi/Documents/GitHub/Fenicotteri-equilibristi'
db_path=root_path+'/Database/'
fig_path=root_path+'/Figures/'

df_FE=pd.DataFrame()
ROI_DK_list=['bankssts L','bankssts R','caudalanteriorcingulate L','caudalanteriorcingulate R','caudalmiddlefrontal L','caudalmiddlefrontal R','cuneus L','cuneus R','entorhinal L','entorhinal R','frontalpole L','frontalpole R','fusiform L','fusiform R','inferiorparietal L','inferiorparietal R','inferiortemporal L','inferiortemporal R','insula L','insula R','isthmuscingulate L','isthmuscingulate R','lateraloccipital L','lateraloccipital R','lateralorbitofrontal L','lateralorbitofrontal R','lingual L','lingual R','medialorbitofrontal L','medialorbitofrontal R','middletemporal L','middletemporal R','paracentral L','paracentral R','parahippocampal L','parahippocampal R','parsopercularis L','parsopercularis R','parsorbitalis L','parsorbitalis R','parstriangularis L','parstriangularis R','pericalcarine L','pericalcarine R','postcentral L','postcentral R','posteriorcingulate L','posteriorcingulate R','precentral L','precentral R','precuneus L','precuneus R','rostralanteriorcingulate L','rostralanteriorcingulate R','rostralmiddlefrontal L','rostralmiddlefrontal R','superiorfrontal L','superiorfrontal R','superiorparietal L','superiorparietal R','superiortemporal L','superiortemporal R','supramarginal L','supramarginal R','temporalpole L','temporalpole R','transversetemporal L','transversetemporal R']
ROIs = '%0d'.join(ROI_DK_list)
Region=['LT','RT','LL','RL','LF','RF','LO','RO','LT','RT','LPF','RPF','LT','RT','LP','RP','LT','RT','LT','RT','LL','RL','LO','RO','LPF','RPF','LO','RO','LPF','RPF','LT','RT','LC','RC','LT','RT','LF','RF','LPF','RPF','LF','RF','LO','RO','LC','RC','LL','RL','LC','RC','LP','RP','LL','RL','LF','RF','LF','RF','LP','RP','LT','RT','LP','RP','LT','RT','LT','RT']

#%% plot reliable results - March 2022
import mat73
p_thresh=0.05
#f_name='Reliable_Analysis_' + str( p_thresh ) + '_Group.mat'
f_name = "Reliable_Analysis_Edge_Based_BH_p_thresh_edges_0.05_p_thresh_nodes_0.05_Group.mat"
data_dict = mat73.loadmat (db_path+f_name)
sig_abs_diff_val=data_dict['signif_abs_cmp']
sig_abs_diff_val=sig_abs_diff_val.astype ( int )
sig_diff_val = data_dict[ 'signif_cmp' ]
sig_diff_val = sig_diff_val.astype ( int )

# fig_abs_reliab = plt.figure ( figsize=(6 , 6) , facecolor='white' )
# mne.viz.plot_connectivity_circle ( sig_abs_diff_val , node_names=ROI_DK_list ,
#                                        colorbar_pos=(0.3 , -0.1) ,
#                                        fontsize_names=5.4 , fontsize_title=9 ,
#                                        colormap='Greys', facecolor='white', textcolor='black',
#                                        node_edgecolor='white',
#                                        title='p-values < 0.05, BH corrected, reliability' ,
#                                        fig=fig_abs_reliab )
# fig_abs_reliab.savefig ( fig_path + 'Connect_Reliable_Analysis_abs_diff_Edge_Based_BH_p_thresh_edges_0.05_p_thresh_nodes_0.05_Group' + '.png' , dpi=600 ,
#                    facecolor='white' )

fig_reliab = plt.figure ( figsize=(12 ,12) , facecolor='white' )
mne.viz.plot_connectivity_circle ( sig_diff_val , node_names=ROI_DK_list ,
                                       colorbar=False,
                                       #colorbar_pos=(0.3 , -0.1) ,
                                       fontsize_names=21 , #fontsize_title=9 ,
                                       colormap='Greys', facecolor='white', textcolor='black',
                                       node_edgecolor='white',
                                       #title='p-values < 0.05, BH corrected, reliability' ,
                                       fig=fig_reliab )
fig_reliab.savefig ( fig_path + 'Connect_Reliable_Analysis_diff_Edge_Based_BH_p_thresh_edges_0.05_p_thresh_nodes_0.05_Group' + '.png' , dpi=600 ,
                   facecolor='white' )


#%% correlation from pre-selected ROIs via BH
f='Correlation_Analysis_Edge_Based_BH_Group_OptPerm.mat'
temp2=scipy.io.loadmat(db_path+f)

r_val_diff=temp2['rho_edges_diff_thresholded']
r_val_diff[np.isnan(r_val_diff)] = 0
r_val_abs_diff=temp2['rho_edges_abs_diff_thresholded']
r_val_abs_diff[np.isnan(r_val_abs_diff)] = 0
pval_diff=temp2['pval_edges_diff_thresholded']
pval_abs_diff=temp2['pval_edges_abs_diff_thresholded']

max=np.max(np.abs((r_val_abs_diff)))
fig_abs_corr = plt.figure(figsize=(6, 6), facecolor='white')
mne.viz.plot_connectivity_circle(r_val_abs_diff, node_names=ROI_DK_list,
                                         colorbar_pos=(0.3 , -0.1) ,
                                         colormap='RdBu_r' ,
                                         facecolor='white', textcolor='black',
                                         node_edgecolor='white',
                                         vmin=-max, vmax=max,
                                         fontsize_names=5.4 , fontsize_title=9 ,
                                         title='r-values, correlation w/ BCI scores' ,
                                         fig=fig_abs_corr)
fig_abs_corr.savefig(fig_path+'Connect_Correlation_Analysis_BH_Group_OptPerm_R_Values_Edges_abs_Diff_thresholded.pdf', dpi=300, facecolor='white')


max=np.max(np.abs((r_val_diff)))
fig_corr = plt.figure(figsize=(6, 6), facecolor='white')
mne.viz.plot_connectivity_circle(r_val_diff, node_names=ROI_DK_list,
                                         colorbar_pos=(0.3 , -0.1) ,
                                         colormap='RdBu_r' ,
                                         facecolor='white', textcolor='black',
                                         node_edgecolor='white',
                                         vmin=-max, vmax=max,
                                         fontsize_names=5.4 , fontsize_title=9 ,
                                         title='r-values, correlation w/ BCI scores (p<0.05)' ,
                                         fig=fig_corr)
fig_corr.savefig(fig_path+'Connect_Correlation_Analysis_BH_Group_OptPerm_R_Values_Edges_Diff_thresholded.pdf', dpi=300, facecolor='white')

#%% prova 01/04/2022 diversi modi di pre-selezioni degli edges
f='Correlation_Analysis_Edge_Based_Nodi_Group_OptPerm_prova.mat'
temp2=scipy.io.loadmat(db_path+f)

r_val_diff=temp2['HHH_r_thresholded']
r_val_diff[np.isnan(r_val_diff)] = 0
r_val_abs_diff=temp2['HHH_abs_r_thresholded']
r_val_abs_diff[np.isnan(r_val_abs_diff)] = 0
pval_diff=temp2['HHH_p_thresholded']
pval_abs_diff=temp2['HHH_abs_p_thresholded']

max=np.max(np.abs((r_val_abs_diff)))
fig_abs_corr = plt.figure(figsize=(6, 6), facecolor='white')
mne.viz.plot_connectivity_circle(r_val_abs_diff, node_names=ROI_DK_list,
                                         colorbar_pos=(0.3 , -0.1) ,
                                         colormap='RdBu_r' ,
                                         facecolor='white', textcolor='black',
                                         node_edgecolor='white',
                                         vmin=-max, vmax=max,
                                         fontsize_names=5.4 , fontsize_title=9 ,
                                         title='r-values, correlation w/ BCI scores' ,
                                         fig=fig_abs_corr)
fig_abs_corr.savefig(fig_path+'Connect_Correlation_Analysis_PROVA_Group_OptPerm_R_Values_Edges_abs_Diff_thresholded.pdf', dpi=300, facecolor='white')


max=np.max(np.abs((r_val_diff)))
fig_corr = plt.figure(figsize=(6, 6), facecolor='white')
mne.viz.plot_connectivity_circle(r_val_diff, node_names=ROI_DK_list,
                                         colorbar_pos=(0.3 , -0.1) ,
                                         colormap='RdBu_r' ,
                                         facecolor='white', textcolor='black',
                                         node_edgecolor='white',
                                         vmin=-max, vmax=max,
                                         fontsize_names=5.4 , fontsize_title=9 ,
                                         title='r-values, correlation w/ BCI scores (p<0.05)' ,
                                         fig=fig_corr)
fig_corr.savefig(fig_path+'Connect_Correlation_Analysis_PROVA_Group_OptPerm_R_Values_Edges_Diff_thresholded.pdf', dpi=300, facecolor='white')


#%% test - threshold on r-val
r_thresh=0.5#0.6#0.45#0.3

f='Correlation_Analysis_Edge_Based_Nodi_Group_OptPerm_prova_' + str ( r_thresh ) + '.mat'
temp2=scipy.io.loadmat(db_path+f)

r_val_diff=temp2['HHH_r_thresholded_2']
r_val_diff[np.isnan(r_val_diff)] = 0
r_val_abs_diff=temp2['HHH_abs_r_thresholded_2']
r_val_abs_diff[np.isnan(r_val_abs_diff)] = 0
pval_diff=temp2['HHH_p_thresholded_2']
pval_abs_diff=temp2['HHH_abs_p_thresholded_2']

max=np.max(np.abs((r_val_abs_diff)))
fig_abs_corr = plt.figure(figsize=(6, 6), facecolor='white')
mne.viz.plot_connectivity_circle(r_val_abs_diff, node_names=ROI_DK_list,
                                         colorbar_pos=(0.3 , -0.1) ,
                                         colormap='RdBu_r' ,
                                         facecolor='white', textcolor='black',
                                         node_edgecolor='white',
                                         vmin=-max, vmax=max,
                                         fontsize_names=5.4 , fontsize_title=9 ,
                                         title='r-values, correlation w/ BCI scores' ,
                                         fig=fig_abs_corr)
fig_abs_corr.savefig(fig_path+'Connect_Correlation_Analysis_PROVA_Group_OptPerm_R_Values_Edges_abs_Diff_thresholded_r' + str ( r_thresh ) + '.pdf' , dpi=300, facecolor='white')


max=np.max(np.abs((r_val_diff)))
fig_corr = plt.figure(figsize=(6, 6), facecolor='white')
mne.viz.plot_connectivity_circle(r_val_diff, node_names=ROI_DK_list,
                                         colorbar_pos=(0.3 , -0.1) ,
                                         colormap='RdBu_r' ,
                                         facecolor='white', textcolor='black',
                                         node_edgecolor='white',
                                         vmin=-max, vmax=max,
                                         fontsize_names=5.4 , fontsize_title=9 ,
                                         title='r-values, correlation w/ BCI scores (p<0.05)' ,
                                         fig=fig_corr)
fig_corr.savefig(fig_path+'Connect_Correlation_Analysis_PROVA_Group_OptPerm_R_Values_Edges_Diff_thresholded_r' + str ( r_thresh ) + '.pdf' , dpi=300, facecolor='white')


#%% reboot correlation - BCI - post cleaning
f='Correlation_Analysis_BCI_Based_Nodi_Group_OptPerm_Paper.mat'
temp2=scipy.io.loadmat(db_path+f)

r_val_diff=temp2['matr_edges_diff_corr_BCI_r_thresholded']
r_val_diff[np.isnan(r_val_diff)] = 0
pval_diff=temp2['matr_edges_diff_corr_BCI_p_thresholded']


max=np.max(np.abs((r_val_diff)))
fig_corr = plt.figure(figsize=(12,12), facecolor='white')
mne.viz.plot_connectivity_circle(r_val_diff, node_names=ROI_DK_list,
                                         #colorbar_pos=(0.81 , -0.1) ,
                                         colorbar=False,
                                         colormap='RdBu_r' ,
                                         facecolor='white', textcolor='black',
                                         node_edgecolor='white',
                                         vmin=-max, vmax=max,
                                         fontsize_names=21 , #fontsize_title=9 ,
                                         #title='r-values, correlation w/ BCI scores (p<0.05)' ,
                                         fig=fig_corr)
fig_corr.savefig(fig_path+'Connect_Correlation_Analysis_Group_OptPerm_R_Values_Edges_Diff_thresholded_0,05.png', dpi=600, facecolor='white')

#%% Rosenberg
f='Correlation_Analysis_Rosenberg_Based_Nodi_Group_OptPerm.mat'
temp2=scipy.io.loadmat(db_path+f)

r_val_diff=temp2['matr_edges_diff_corr_Rosenberg_r_thresholded']
r_val_diff[np.isnan(r_val_diff)] = 0
r_val_abs_diff=temp2['matr_edges_abs_diff_corr_Rosenberg_r_thresholded']
r_val_abs_diff[np.isnan(r_val_abs_diff)] = 0
pval_diff=temp2['matr_edges_diff_corr_Rosenberg_p_thresholded']
pval_abs_diff=temp2['matr_edges_abs_diff_corr_Rosenberg_p_thresholded']

max=np.max(np.abs((r_val_abs_diff)))
fig_abs_corr = plt.figure(figsize=(6, 6), facecolor='white')
mne.viz.plot_connectivity_circle(r_val_abs_diff, node_names=ROI_DK_list,
                                         colorbar_pos=(0.3 , -0.1) ,
                                         colormap='RdBu_r' ,
                                         facecolor='white', textcolor='black',
                                         node_edgecolor='white',
                                         vmin=-max, vmax=max,
                                         fontsize_names=5.4 , fontsize_title=9 ,
                                         title='r-values, correlation w/ Rosenberg scores' ,
                                         fig=fig_abs_corr)
fig_abs_corr.savefig(fig_path+'Connect_Correlation_Analysis_Rosenberg_Group_OptPerm_R_Values_Edges_abs_Diff_thresholded_0,05.pdf', dpi=300, facecolor='white')


max=np.max(np.abs((r_val_diff)))
fig_corr = plt.figure(figsize=(6, 6), facecolor='white')
mne.viz.plot_connectivity_circle(r_val_diff, node_names=ROI_DK_list,
                                         colorbar_pos=(0.3 , -0.1) ,
                                         colormap='RdBu_r' ,
                                         facecolor='white', textcolor='black',
                                         node_edgecolor='white',
                                         vmin=-max, vmax=max,
                                         fontsize_names=5.4 , fontsize_title=9 ,
                                         title='r-values, correlation w/ Rosenberg scores (p<0.05)' ,
                                         fig=fig_corr)
fig_corr.savefig(fig_path+'Connect_Correlation_Analysis_Rosenberg_Group_OptPerm_R_Values_Edges_Diff_thresholded_0,05.pdf', dpi=300, facecolor='white')

#%% STAI-YA - 0 signif
f='Correlation_Analysis_STAI_YA_Based_Nodi_Group_OptPerm.mat'
temp2=scipy.io.loadmat(db_path+f)

r_val_diff=temp2['matr_edges_diff_corr_STAI_YA_r_thresholded']
r_val_diff[np.isnan(r_val_diff)] = 0
r_val_abs_diff=temp2['matr_edges_abs_diff_corr_STAI_YA_r_thresholded']
r_val_abs_diff[np.isnan(r_val_abs_diff)] = 0
pval_diff=temp2['matr_edges_diff_corr_STAI_YA_p_thresholded']
pval_abs_diff=temp2['matr_edges_abs_diff_corr_STAI_YA_p_thresholded']

max=np.max(np.abs((r_val_abs_diff)))
fig_abs_corr = plt.figure(figsize=(6, 6), facecolor='white')
mne.viz.plot_connectivity_circle(r_val_abs_diff, node_names=ROI_DK_list,
                                         colorbar_pos=(0.3 , -0.1) ,
                                         colormap='RdBu_r' ,
                                         facecolor='white', textcolor='black',
                                         node_edgecolor='white',
                                         vmin=-max, vmax=max,
                                         fontsize_names=5.4 , fontsize_title=9 ,
                                         title='r-values, correlation w/ STAI_YA scores' ,
                                         fig=fig_abs_corr)
fig_abs_corr.savefig(fig_path+'Connect_Correlation_Analysis_STAI_YA_Group_OptPerm_R_Values_Edges_abs_Diff_thresholded_0,05.pdf', dpi=300, facecolor='white')


max=np.max(np.abs((r_val_diff)))
fig_corr = plt.figure(figsize=(6, 6), facecolor='white')
mne.viz.plot_connectivity_circle(r_val_diff, node_names=ROI_DK_list,
                                         colorbar_pos=(0.3 , -0.1) ,
                                         colormap='RdBu_r' ,
                                         facecolor='white', textcolor='black',
                                         node_edgecolor='white',
                                         vmin=-max, vmax=max,
                                         fontsize_names=5.4 , fontsize_title=9 ,
                                         title='r-values, correlation w/ STAI_YA scores (p<0.05)' ,
                                         fig=fig_corr)
fig_corr.savefig(fig_path+'Connect_Correlation_Analysis_STAI_YA_Group_OptPerm_R_Values_Edges_Diff_thresholded_0,05.pdf', dpi=300, facecolor='white')

#%% Kinaesth
f='Correlation_Analysis_Kinaesth_Based_Nodi_Group_OptPerm.mat'
temp2=scipy.io.loadmat(db_path+f)

r_val_diff=temp2['matr_edges_diff_corr_Kinaesth_r_thresholded']
r_val_diff[np.isnan(r_val_diff)] = 0
r_val_abs_diff=temp2['matr_edges_abs_diff_corr_Kinaesth_r_thresholded']
r_val_abs_diff[np.isnan(r_val_abs_diff)] = 0
pval_diff=temp2['matr_edges_diff_corr_Kinaesth_p_thresholded']
pval_abs_diff=temp2['matr_edges_abs_diff_corr_Kinaesth_p_thresholded']

max=np.max(np.abs((r_val_abs_diff)))
fig_abs_corr = plt.figure(figsize=(6, 6), facecolor='white')
mne.viz.plot_connectivity_circle(r_val_abs_diff, node_names=ROI_DK_list,
                                         colorbar_pos=(0.3 , -0.1) ,
                                         colormap='RdBu_r' ,
                                         facecolor='white', textcolor='black',
                                         node_edgecolor='white',
                                         vmin=-max, vmax=max,
                                         fontsize_names=5.4 , fontsize_title=9 ,
                                         title='r-values, correlation w/ Kinaesth scores' ,
                                         fig=fig_abs_corr)
fig_abs_corr.savefig(fig_path+'Connect_Correlation_Analysis_Kinaesth_Group_OptPerm_R_Values_Edges_abs_Diff_thresholded_0,05.pdf', dpi=300, facecolor='white')


max=np.max(np.abs((r_val_diff)))
fig_corr = plt.figure(figsize=(6, 6), facecolor='white')
mne.viz.plot_connectivity_circle(r_val_diff, node_names=ROI_DK_list,
                                         colorbar_pos=(0.3 , -0.1) ,
                                         colormap='RdBu_r' ,
                                         facecolor='white', textcolor='black',
                                         node_edgecolor='white',
                                         vmin=-max, vmax=max,
                                         fontsize_names=5.4 , fontsize_title=9 ,
                                         title='r-values, correlation w/ Kinaesth scores (p<0.05)' ,
                                         fig=fig_corr)
fig_corr.savefig(fig_path+'Connect_Correlation_Analysis_Kinaesth_Group_OptPerm_R_Values_Edges_Diff_thresholded_0,05.pdf', dpi=300, facecolor='white')

#%% IntVis

f='Correlation_Analysis_IntVis_Based_Nodi_Group_OptPerm.mat'
temp2=scipy.io.loadmat(db_path+f)

r_val_diff=temp2['matr_edges_diff_corr_IntVis_r_thresholded']
r_val_diff[np.isnan(r_val_diff)] = 0
r_val_abs_diff=temp2['matr_edges_abs_diff_corr_IntVis_r_thresholded']
r_val_abs_diff[np.isnan(r_val_abs_diff)] = 0
pval_diff=temp2['matr_edges_diff_corr_IntVis_p_thresholded']
pval_abs_diff=temp2['matr_edges_abs_diff_corr_IntVis_p_thresholded']

max=np.max(np.abs((r_val_abs_diff)))
fig_abs_corr = plt.figure(figsize=(6, 6), facecolor='white')
mne.viz.plot_connectivity_circle(r_val_abs_diff, node_names=ROI_DK_list,
                                         colorbar_pos=(0.3 , -0.1) ,
                                         colormap='RdBu_r' ,
                                         facecolor='white', textcolor='black',
                                         node_edgecolor='white',
                                         vmin=-max, vmax=max,
                                         fontsize_names=5.4 , fontsize_title=9 ,
                                         title='r-values, correlation w/ IntVis scores' ,
                                         fig=fig_abs_corr)
fig_abs_corr.savefig(fig_path+'Connect_Correlation_Analysis_IntVis_Group_OptPerm_R_Values_Edges_abs_Diff_thresholded_0,05.pdf', dpi=300, facecolor='white')


max=np.max(np.abs((r_val_diff)))
fig_corr = plt.figure(figsize=(6, 6), facecolor='white')
mne.viz.plot_connectivity_circle(r_val_diff, node_names=ROI_DK_list,
                                         colorbar_pos=(0.3 , -0.1) ,
                                         colormap='RdBu_r' ,
                                         facecolor='white', textcolor='black',
                                         node_edgecolor='white',
                                         vmin=-max, vmax=max,
                                         fontsize_names=5.4 , fontsize_title=9 ,
                                         title='r-values, correlation w/ IntVis scores (p<0.05)' ,
                                         fig=fig_corr)
fig_corr.savefig(fig_path+'Connect_Correlation_Analysis_IntVis_Group_OptPerm_R_Values_Edges_Diff_thresholded_0,05.pdf', dpi=300, facecolor='white')

#%% ExtVis
f='Correlation_Analysis_ExtVis_Based_Nodi_Group_OptPerm.mat'
temp2=scipy.io.loadmat(db_path+f)

r_val_diff=temp2['matr_edges_diff_corr_ExtVis_r_thresholded']
r_val_diff[np.isnan(r_val_diff)] = 0
r_val_abs_diff=temp2['matr_edges_abs_diff_corr_ExtVis_r_thresholded']
r_val_abs_diff[np.isnan(r_val_abs_diff)] = 0
pval_diff=temp2['matr_edges_diff_corr_ExtVis_p_thresholded']
pval_abs_diff=temp2['matr_edges_abs_diff_corr_ExtVis_p_thresholded']

max=np.max(np.abs((r_val_abs_diff)))
fig_abs_corr = plt.figure(figsize=(6, 6), facecolor='white')
mne.viz.plot_connectivity_circle(r_val_abs_diff, node_names=ROI_DK_list,
                                         colorbar_pos=(0.3 , -0.1) ,
                                         colormap='RdBu_r' ,
                                         facecolor='white', textcolor='black',
                                         node_edgecolor='white',
                                         vmin=-max, vmax=max,
                                         fontsize_names=5.4 , fontsize_title=9 ,
                                         title='r-values, correlation w/ ExtVis scores' ,
                                         fig=fig_abs_corr)
fig_abs_corr.savefig(fig_path+'Connect_Correlation_Analysis_ExtVis_Group_OptPerm_R_Values_Edges_abs_Diff_thresholded_0,05.pdf', dpi=300, facecolor='white')


max=np.max(np.abs((r_val_diff)))
fig_corr = plt.figure(figsize=(6, 6), facecolor='white')
mne.viz.plot_connectivity_circle(r_val_diff, node_names=ROI_DK_list,
                                         colorbar_pos=(0.3 , -0.1) ,
                                         colormap='RdBu_r' ,
                                         facecolor='white', textcolor='black',
                                         node_edgecolor='white',
                                         vmin=-max, vmax=max,
                                         fontsize_names=5.4 , fontsize_title=9 ,
                                         title='r-values, correlation w/ ExtVis scores (p<0.05)' ,
                                         fig=fig_corr)
fig_corr.savefig(fig_path+'Connect_Correlation_Analysis_ExtVis_Group_OptPerm_R_Values_Edges_Diff_thresholded_0,05.pdf', dpi=300, facecolor='white')
