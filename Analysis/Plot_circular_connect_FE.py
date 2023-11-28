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

list_p_thresh=[0.01, 0.025, 0.05]
for kk_p_thresh, p_thresh in enumerate(list_p_thresh):

    f_name='Reliable_Analysis_' + str( p_thresh ) + '_Group.mat'
    data_dict = mat73.loadmat (db_path+f_name)
    sig_abs_diff_val=data_dict['signif_abs_diff']
    sig_abs_diff_val=sig_abs_diff_val.astype ( int )
    sig_diff_val = data_dict[ 'signif_diff' ]
    sig_diff_val = sig_diff_val.astype ( int )

    fig2 = plt.figure ( figsize=(6 , 6) , facecolor='black' )
    mne.viz.plot_connectivity_circle ( sig_abs_diff_val , node_names=ROI_DK_list ,
                                       colorbar_pos=(0.3 , -0.1) ,
                                       fontsize_names=5.4 , fontsize_title=9 ,
                                       title='p-values, reliability' ,
                                       fig=fig2 )
    fig2.savefig ( fig_path + 'Reliable_p_Values_abs_diff_Edges_p_thresh_' + str ( p_thresh ) + '.pdf' , dpi=300 ,
                   facecolor='black' )

    fig3 = plt.figure ( figsize=(6 , 6) , facecolor='black' )
    mne.viz.plot_connectivity_circle ( sig_diff_val , node_names=ROI_DK_list ,
                                       colorbar_pos=(0.3 , -0.1) ,
                                       fontsize_names=5.4 , fontsize_title=9 ,
                                       title='p-values, reliability' ,
                                       fig=fig3 )
    fig3.savefig ( fig_path + 'Reliable_p_Values_diff_Edges_p_thresh_' + str ( p_thresh ) + '.pdf' , dpi=300 ,
                   facecolor='black' )

#%% plot correlation results - March 2022
p_thresh=0.05
f='Correlation_Analysis_'+str(p_thresh)+'_Group.mat'
temp2=scipy.io.loadmat(db_path+f)
r_val=temp2['r_plot']
p_val=temp2['p_plot']
r_mat = r_val
p_mat = p_val

## TODO: debug, like a mask that hides things
# fig = plt.figure(figsize=(6, 6), facecolor='black')
# mne.viz.plot_connectivity_circle(p_mat, node_names=ROI_DK_list,
#                                 colorbar_pos=(0.3, -0.1),
#                                 colormap='YlOrBr',
#                                 vmin=0, vmax=0.05,
#                                 fontsize_names=5.4, fontsize_title=9,
#                                 title='p-values, correlation w/ BCI scores',
#                                     fig=fig)
# fig.savefig(fig_path+'Corr_p_Values_Edges_p_thresh_'+str(p_thresh)+'.pdf', dpi=300)#, facecolor='black')

fig2 = plt.figure(figsize=(6, 6), facecolor='black')
max=np.max(np.abs((r_mat)))
mne.viz.plot_connectivity_circle(r_mat, node_names=ROI_DK_list,
                                         colorbar_pos=(0.3 , -0.1) ,
                                         colormap='twilight' ,
                                         vmin=-max, vmax=max,
                                         fontsize_names=5.4 , fontsize_title=9 ,
                                         title='r-values, correlation w/ BCI scores' ,
                                         fig=fig2)
fig2.savefig(fig_path+'Corr_R_Values_Edges_p_thresh_'+str(p_thresh)+'.pdf', dpi=300, facecolor='black')
#%% 10k
p_thresh=0.05#0.01
f='Correlation_Analysis_Bonfe_Group_OptPerm'
temp2=scipy.io.loadmat(db_path+f)
r_val=temp2['rho_edges_diff_plot']
p_val=temp2['pval_edges_diff_plot']
r_mat = r_val
p_mat = p_val

## TODO: debug, like a mask that hides things
fig2 = plt.figure(figsize=(6, 6), facecolor='black')
max=np.max(np.abs((r_mat)))
mne.viz.plot_connectivity_circle(r_mat, node_names=ROI_DK_list,
                                         colorbar_pos=(0.3 , -0.1) ,
                                         colormap='twilight' ,
                                         vmin=-max, vmax=max,
                                         fontsize_names=5.4 , fontsize_title=9 ,
                                         title='r-values, correlation w/ BCI scores' ,
                                         fig=fig2)
fig2.savefig(fig_path+'Correlation_Analysis_Bonfe_Group_OptPerm_R_Values_Edges_Diff_p_thresh_'+str(p_thresh)+'.pdf', dpi=300, facecolor='black')

r_val=temp2['rho_edges_abs_diff_plot']
p_val=temp2['pval_edges_abs_diff_plot']
r_mat = r_val
p_mat = p_val

fig2 = plt.figure(figsize=(6, 6), facecolor='black')
max=np.max(np.abs((r_mat)))
mne.viz.plot_connectivity_circle(r_mat, node_names=ROI_DK_list,
                                         colorbar_pos=(0.3 , -0.1) ,
                                         colormap='twilight' ,
                                         vmin=-max, vmax=max,
                                         fontsize_names=5.4 , fontsize_title=9 ,
                                         title='r-values, correlation w/ BCI scores' ,
                                         fig=fig2)
fig2.savefig(fig_path+'Correlation_Analysis_Bonfe_Group_OptPerm_R_Values_Edges_abs_Diff_p_thresh_'+str(p_thresh)+'.pdf', dpi=300, facecolor='black')

#%% plot reliable connections - meeting March 2022
#p_thresh=0.05#0.01
f='Reliable_Analysis_Edge_Based_Bonfe_p_thresh_edges_2.2282e-05_p_thresh_nodes_0.00073529_Group'
temp2=scipy.io.loadmat(db_path+f)
signif_diff=temp2['signif_cmp']
signif_abs_diff=temp2['signif_abs_cmp']

## TODO: debug, like a mask that hides things
fig2 = plt.figure(figsize=(6, 6), facecolor='black')
max=np.max(np.abs((signif_diff)))
mne.viz.plot_connectivity_circle(signif_diff, node_names=ROI_DK_list,
                                         colorbar_pos=(0.3 , -0.1) ,
                                         colormap='hot' ,
                                         vmin=0, vmax=max,
                                         fontsize_names=5.4 , fontsize_title=9 ,
                                         title='p-values, reliability, group, MI vs Rest' ,
                                         fig=fig2)
fig2.savefig(fig_path+'Connectome_Reliable_Analysis_Edge_Based_Bonfe_p_thresh_edges_2.2282e-05_p_thresh_nodes_0.00073529_Group_diff.pdf', dpi=300, facecolor='black')

fig2 = plt.figure(figsize=(6, 6), facecolor='black')
max=np.max(np.abs((signif_abs_diff)))
mne.viz.plot_connectivity_circle(signif_abs_diff, node_names=ROI_DK_list,
                                         colorbar_pos=(0.3 , -0.1) ,
                                         colormap='hot' ,
                                         vmin=0, vmax=max,
                                         fontsize_names=5.4 , fontsize_title=9 ,
                                         title='p-values, reliability, group, MI vs Rest' ,
                                         fig=fig2)
fig2.savefig(fig_path+'Connectome_Reliable_Analysis_Edge_Based_Bonfe_p_thresh_edges_2.2282e-05_p_thresh_nodes_0.00073529_Group_abs_diff.pdf', dpi=300, facecolor='black')

#%% correlation from pre-selected ROIs via BHFDR
f='Correlation_Analysis_Edge_Based_BHFDR_Group_OptPerm.mat'
temp2=scipy.io.loadmat(db_path+f)

r_val_diff=temp2['rho_edges_diff_plot']
r_val_abs_diff=temp2['rho_edges_abs_diff_plot']
pval_diff=temp2['pval_edges_diff_plot']
pval_abs_diff=temp2['pval_edges_abs_diff_plot']

max=np.max(np.abs((r_val_diff)))
fig3 = plt.figure(figsize=(6, 6), facecolor='black')
mne.viz.plot_connectivity_circle(r_val_diff, node_names=ROI_DK_list,
                                         colorbar_pos=(0.3 , -0.1) ,
                                         colormap='twilight' ,
                                         vmin=-max, vmax=max,
                                         fontsize_names=5.4 , fontsize_title=9 ,
                                         title='r-values, correlation w/ BCI scores' ,
                                         fig=fig3)
fig3.savefig(fig_path+'Correlation_Analysis_BHFDR_Group_OptPerm_R_Values_Edges_Diff.pdf', dpi=300, facecolor='black')


max=np.max(np.abs((r_val_abs_diff)))
fig4 = plt.figure(figsize=(6, 6), facecolor='black')
mne.viz.plot_connectivity_circle(r_val_abs_diff, node_names=ROI_DK_list,
                                         colorbar_pos=(0.3 , -0.1) ,
                                         colormap='twilight' ,
                                         vmin=-max, vmax=max,
                                         fontsize_names=5.4 , fontsize_title=9 ,
                                         title='r-values, correlation w/ BCI scores' ,
                                         fig=fig4)
fig4.savefig(fig_path+'Correlation_Analysis_BHFDR_Group_OptPerm_R_Values_Edges_abs_Diff.pdf', dpi=300, facecolor='black')
