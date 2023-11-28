# TODO: https://towardsdatascience.com/chord-diagrams-of-protein-interaction-networks-in-python-9589affc8b91
 # https://github.com/ericmjl/nxviz/blob/master/examples/circos/group_labels.py
# create a script from a matrix and a vector of labels - plot connectome
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import mat73
import os
from nxviz import CircosPlot
import scipy.io
import numpy as np
#%%
root_path='/Users/marieconstance.corsi/Documents/GitHub/Fenicotteri-equilibristi'
db_path=root_path+'/Database/'
fig_path=root_path+'/Figures/'

df_FE=pd.DataFrame()
ROI_DK_list=['bankssts L','bankssts R','caudalanteriorcingulate L','caudalanteriorcingulate R','caudalmiddlefrontal L','caudalmiddlefrontal R','cuneus L','cuneus R','entorhinal L','entorhinal R','frontalpole L','frontalpole R','fusiform L','fusiform R','inferiorparietal L','inferiorparietal R','inferiortemporal L','inferiortemporal R','insula L','insula R','isthmuscingulate L','isthmuscingulate R','lateraloccipital L','lateraloccipital R','lateralorbitofrontal L','lateralorbitofrontal R','lingual L','lingual R','medialorbitofrontal L','medialorbitofrontal R','middletemporal L','middletemporal R','paracentral L','paracentral R','parahippocampal L','parahippocampal R','parsopercularis L','parsopercularis R','parsorbitalis L','parsorbitalis R','parstriangularis L','parstriangularis R','pericalcarine L','pericalcarine R','postcentral L','postcentral R','posteriorcingulate L','posteriorcingulate R','precentral L','precentral R','precuneus L','precuneus R','rostralanteriorcingulate L','rostralanteriorcingulate R','rostralmiddlefrontal L','rostralmiddlefrontal R','superiorfrontal L','superiorfrontal R','superiorparietal L','superiorparietal R','superiortemporal L','superiortemporal R','supramarginal L','supramarginal R','temporalpole L','temporalpole R','transversetemporal L','transversetemporal R']
ROIs = '%0d'.join(ROI_DK_list)
Region=['LT','RT','LL','RL','LF','RF','LO','RO','LT','RT','LPF','RPF','LT','RT','LP','RP','LT','RT','LT','RT','LL','RL','LO','RO','LPF','RPF','LO','RO','LPF','RPF','LT','RT','LC','RC','LT','RT','LF','RF','LPF','RPF','LF','RF','LO','RO','LC','RC','LL','RL','LC','RC','LP','RP','LL','RL','LF','RF','LF','RF','LP','RP','LT','RT','LP','RP','LT','RT','LT','RT']

#%% load files associated with the individual Edge_Analysis
files = [i for i in os.listdir(db_path) if os.path.isfile(os.path.join(db_path,i)) and \
         'Edge_Analysis_' in i]

for f in files:
    temp=mat73.loadmat(db_path+f)
    matrix_pval=temp['matrix_pval_diff_edge_bonf_correct']
    # per mettere 1 agli edges significativo, 0 senno
    matrix_pval[ matrix_pval == 1 ] = 2
    matrix_pval[ matrix_pval == 0 ] = 1
    matrix_pval[ matrix_pval == 2 ] = 0
    adj_mat = matrix_pval
    G = nx.from_numpy_matrix ( adj_mat , create_using=nx.Graph , parallel_edges=False )
    mapping = dict ()
    region = dict ()
    keys = range ( 68 )
    for kk_ROI in keys:
        mapping[ kk_ROI ] = ROI_DK_list[ kk_ROI ]
        region[ ROI_DK_list[ kk_ROI ] ] = Region[ kk_ROI ]

    G = nx.relabel_nodes ( G , mapping )

    for n , d in G.nodes ( data=True ):
        G.node[ n ][ "class" ] = region[ n ]

    weights = pd.Series (
        {tuple ( edge_data[ :-1 ] ): edge_data[ -1 ][ "weight" ] for edge_data in G.edges ( data=True )} )

    c = CircosPlot (
        G ,
        fontsize=6 ,
        nodeprops={"radius": 3} ,
        node_grouping="class" ,
        node_color="class" ,
        node_order="class" ,
        node_labels=True ,
        node_label_layout="rotation" ,
        group_label_position="middle" ,
        group_label_color=True ,
        group_label_offset=3 ,
        group_by="group" ,
        node_color_by="group" ,
        edge_color_by="source_node_color" ,
        edge_alpha_by="weight" ,
        edge_width=weights.array
    )
    c.draw ()

    filename=f.replace('.mat', '.pdf')
    plt.savefig (
        fig_path
        + filename ,
        dpi=300 ,
    )

#%% Group analysis - cf media delle TM per ogni soggetto & condizione
f='Edge_GroupAnalysis_Sess4_MEG_DK_1000perm.mat'
temp=mat73.loadmat(db_path+f)
matrix_pval_diff=temp['matrix_pval_diff_edge_bonf_correct']
matrix_pval_abs_diff=temp['matrix_pval_abs_diff_edge_bonf_correct']

# DIFF
# per mettere 1 agli edges significativo, 0 senno
matrix_pval_diff[ matrix_pval_diff == 1 ] = 2
matrix_pval_diff[ matrix_pval_diff == 0 ] = 1
matrix_pval_diff[ matrix_pval_diff == 2 ] = 0
adj_mat = matrix_pval_diff
G = nx.from_numpy_matrix ( adj_mat , create_using=nx.Graph , parallel_edges=False )
mapping = dict ()
region = dict ()
keys = range ( 68 )
for kk_ROI in keys:
    mapping[ kk_ROI ] = ROI_DK_list[ kk_ROI ]
    region[ ROI_DK_list[ kk_ROI ] ] = Region[ kk_ROI ]

G = nx.relabel_nodes ( G , mapping )

for n , d in G.nodes ( data=True ):
    G.node[ n ][ "class" ] = region[ n ]

weights = pd.Series (
    {tuple ( edge_data[ :-1 ] ): edge_data[ -1 ][ "weight" ] for edge_data in G.edges ( data=True )} )

c = CircosPlot (
    G ,
    fontsize=6 ,
    nodeprops={"radius": 3} ,
    node_grouping="class" ,
    node_color="class" ,
    node_order="class" ,
    node_labels=True ,
    node_label_layout="rotation" ,
    group_label_position="middle" ,
    group_label_color=True ,
    group_label_offset=3 ,
    group_by="group" ,
    node_color_by="group" ,
    edge_color_by="source_node_color" ,
    edge_alpha_by="weight" ,
    edge_width=weights.array
)
c.draw ()

plt.savefig (
    fig_path
    + 'Diff_Edge_GroupAnalysis_Sess4_MEG_DK_1000perm.pdf' ,
    dpi=300 ,
)

# ABS DIFF
# per mettere 1 agli edges significativo, 0 senno
matrix_pval_abs_diff[ matrix_pval_abs_diff == 1 ] = 2
matrix_pval_abs_diff[ matrix_pval_abs_diff == 0 ] = 1
matrix_pval_abs_diff[ matrix_pval_abs_diff == 2 ] = 0
adj_mat = matrix_pval_abs_diff
G = nx.from_numpy_matrix ( adj_mat , create_using=nx.Graph , parallel_edges=False )
mapping = dict ()
region = dict ()
keys = range ( 68 )
for kk_ROI in keys:
    mapping[ kk_ROI ] = ROI_DK_list[ kk_ROI ]
    region[ ROI_DK_list[ kk_ROI ] ] = Region[ kk_ROI ]

G = nx.relabel_nodes ( G , mapping )

for n , d in G.nodes ( data=True ):
    G.node[ n ][ "class" ] = region[ n ]

weights = pd.Series (
    {tuple ( edge_data[ :-1 ] ): edge_data[ -1 ][ "weight" ] for edge_data in G.edges ( data=True )} )

c = CircosPlot (
    G ,
    fontsize=6 ,
    nodeprops={"radius": 3} ,
    node_grouping="class" ,
    node_color="class" ,
    node_order="class" ,
    node_labels=True ,
    node_label_layout="rotation" ,
    group_label_position="middle" ,
    group_label_color=True ,
    group_label_offset=3 ,
    group_by="group" ,
    node_color_by="group" ,
    edge_color_by="source_node_color" ,
    edge_alpha_by="weight" ,
    edge_width=weights.array
)
c.draw ()

plt.savefig (
    fig_path
    + 'Abs_Diff_Edge_GroupAnalysis_Sess4_MEG_DK_1000perm.pdf' ,
    dpi=300 ,
)


#%% plot circular concordance from group
f='Edge_GroupAnalysis_Concordance_Sess4_MEG_DK_1000perm.mat'
temp2=scipy.io.loadmat(db_path+f)
matrix_concord=temp2['concordance']
nb_subj=19
adj_mat = matrix_concord/matrix_concord.max()#-(matrix_concord.min())*np.ones((68,68))#/nb_subj

import mne
fig = plt.figure(figsize=(6, 6), facecolor='black')
mne.viz.plot_connectivity_circle(adj_mat/nb_subj, node_names=ROI_DK_list,
                                colorbar_pos=(0.3, -0.1),
                                vmin=0.5, fontsize_names=5.4, fontsize_title=9,
                                title='Concordance, % of subjects in which signif. links',
                                    fig=fig)#,vmax=matrix_concord.max())
#plt.title('Concordance, group level, links that appears in >50% subjects')
fig.savefig(fig_path+'Concordance.pdf', dpi=300, facecolor='black')


#%% plot circolar - OHBM - reliable edges > 9 subjects
f='relib_9subj.mat'
temp2=scipy.io.loadmat(db_path+f)
relib=temp2['relib']
adj_mat = relib

import mne
fig = plt.figure(figsize=(6, 6), facecolor='black')
mne.viz.plot_connectivity_circle(adj_mat, node_names=ROI_DK_list,
                                colorbar_pos=(0.3, -0.1),
                                fontsize_names=5.4, fontsize_title=9,
                                title='Relib, signif edges in > 9 subjects',
                                    fig=fig)
fig.savefig(fig_path+'Relib.pdf', dpi=300, facecolor='black')
