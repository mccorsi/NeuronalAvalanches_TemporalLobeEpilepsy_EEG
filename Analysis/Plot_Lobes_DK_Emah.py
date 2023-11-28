# %%
"""
==============================================================
Script per raggruppare ROIs in lobi (DK)
===============================================================

"""
# Authors: Marie-Constance Corsi <marie.constance.corsi@gmail.com>
#
# License: BSD (3-clause)

import numpy as np

# %% functions
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

    SubMatrix_M_M = MatrixToBeDivided[np.ix_(idx_M, idx_M)]
    SubMatrix_M_T = MatrixToBeDivided[np.ix_(idx_M, idx_T)]
    SubMatrix_M_P = MatrixToBeDivided[np.ix_(idx_M, idx_P)]
    SubMatrix_M_O = MatrixToBeDivided[np.ix_(idx_M, idx_O)]

    SubMatrix_T_T = MatrixToBeDivided[np.ix_(idx_T, idx_T)]
    SubMatrix_T_P = MatrixToBeDivided[np.ix_(idx_T, idx_P)]
    SubMatrix_T_O = MatrixToBeDivided[np.ix_(idx_T, idx_O)]

    SubMatrix_P_P = MatrixToBeDivided[np.ix_(idx_P, idx_P)]
    SubMatrix_P_O = MatrixToBeDivided[np.ix_(idx_P, idx_O)]

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


# %% parte dalla list dei lobi del atlas - DK
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

#%% raggruppa tutti gli indici associati a un lobo in particolare
idx_DK_postcent = [i for i, j in enumerate(ROI_DK_list) if j == 'postcentral L' or j == 'postcentral R']
idx_DK_caudalmiddlefront = [i for i, j in enumerate(ROI_DK_list) if
                            j == 'caudalmiddlefrontal L' or j == 'caudalmiddlefrontal R']
for i in idx_DK_postcent:
    Region_DK[i] = 'P'
for i in idx_DK_caudalmiddlefront:
    Region_DK[i] = 'M'


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

#%% utile se vuoi attribuire un colore ad ogni lobo
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

# %% esempio del uso della funzione Lobes_Partition che ad ogni matrice attribuisce le sotto-matrici ed la matrice con il nuovo ordine dei indici
MatriceDaDividereInLobi = np.random.random((nb_ROIs_DK, nb_ROIs_DK))
[SubMatrix_F_F, SubMatrix_F_M, SubMatrix_F_T, SubMatrix_F_P, SubMatrix_F_O, \
 SubMatrix_M_M, SubMatrix_M_T, SubMatrix_M_P, SubMatrix_M_O, \
 SubMatrix_T_T, SubMatrix_T_P, SubMatrix_T_O, \
 SubMatrix_P_P, SubMatrix_P_O, \
 SubMatrix_O_O, MatriceDivisaInLobi] = Lobes_Partition(MatriceDaDividereInLobi, idx_F_DK, idx_M_DK, idx_P_DK,
                                                          idx_T_DK, idx_O_DK)
