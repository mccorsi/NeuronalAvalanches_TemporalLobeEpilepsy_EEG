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
