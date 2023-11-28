#%%
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

import pandas as pd
import mat73

from tqdm import tqdm
import pickle
import mne
from mne import create_info, EpochsArray
from mne.decoding import CSP as CSP_MNE

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import ShuffleSplit, cross_val_score

import numpy as np
from scipy.stats import zscore
from moabb.paradigms import MotorImagery


#%%

if os.path.basename(os.getcwd()) == "Fenicotteri-equilibristi":
    os.chdir("Database/0_BCI/Classification/")
if os.path.basename(os.getcwd()) == "py_viz":
    os.chdir("/Users/marieconstance.corsi/Documents/GitHub/Fenicotteri-equilibristi/Database/0_BCI/Classification/")
basedir = os.getcwd()

path_csv_root = os.getcwd() + '/1_Dataset-csv/NETBCI_MEG_DK_Sess4'
if not osp.exists(path_csv_root):
    os.mkdir(path_csv_root)
path_data_root = os.getcwd() + '/3_Dataset-netbci-MEG-sess4-DK'
if not osp.exists(path_data_root):
    os.mkdir(path_data_root)
path_data_root_chan = os.getcwd() + '/3_Dataset-netbci-MEG-sess4-DK'
#%% functions

def transprob(aval,nregions): # (t,r)
    mat = np.zeros((nregions, nregions))
    norm = np.sum(aval, axis=0)
    for t in range(len(aval) - 1):
        ini = np.where(aval[t] == 1)
        mat[ini] += aval[t + 1]
    mat[norm != 0] = mat[norm != 0] / norm[norm != 0][:, None]
    return mat

def Transprob(ZBIN,nregions, val_duration):
    mat = np.zeros((nregions, nregions))
    A = np.sum(ZBIN, axis=1)
    a = np.arange(len(ZBIN))
    idx = np.where(A != 0)[0]
    aout = np.split(a[idx], np.where(np.diff(idx) != 1)[0] + 1)
    ifi = 0
    for iaut in range(len(aout)):
        if len(aout[iaut]) > val_duration:
            mat += transprob(ZBIN[aout[iaut]],nregions)
            ifi += 1
    mat = mat / ifi
    return mat,aout

def threshold_mat(data,thresh=3):
    current_data=data
    binarized_data=np.where(np.abs(current_data)>thresh,1,0)
    return (binarized_data)

def find_avalanches(data,thresh=3, val_duration=2):
    binarized_data=threshold_mat(data,thresh=thresh)
    N=binarized_data.shape[0]
    mat, aout = Transprob(binarized_data.T, N, val_duration)
    aout=np.array(aout,dtype=object)
    list_length=[len(i) for i in aout]
    unique_sizes=set(list_length)
    min_size,max_size=min(list_length),max(list_length)
    list_avalanches_bysize={i:[] for i in unique_sizes}
    for s in aout:
        n=len(s)
        list_avalanches_bysize[n].append(s)
    return(aout,min_size,max_size,list_avalanches_bysize, mat)
#%% load data from matlab
from moabb.datasets.base import BaseDataset
class MEG_DK_Dataset(BaseDataset):
    """
    Dataset from the NETBCI protocol, source space, MEG, DK, Sess4
    """

    def __init__(self):
        super().__init__(
            subjects=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19],
            sessions_per_subject=1, # for the moment...
            events={"right_hand": 1, "rest": 2},
            code="NETBCI_MEG_DK_Dataset",
            interval=[1, 6], # starts when the target is displayed, ends before displaying the result -- longer trials
            paradigm="imagery",
            doi="", # not available yet...
        )

    def _get_single_subject_data(self, subject):
        """return data for a single subject"""
        file_path = self.data_path(subject)

        temp = mat73.loadmat(file_path[0]) # huge files...
        x = temp['Data_concat_moabb'][subject][0] # 2D: nb rois x (nb_trials x nb_samples)
        fs = temp["fs"]
        ch_names = temp['labels_DK'] # actually, roi names here...
        events=temp["Events_moabb"][subject] #1: right_hand, 2:rest

        ch_types = ["eeg" for i in range(np.shape(ch_names)[0])]
        info = mne.create_info(ch_names, fs, ch_types)
        raw = mne.io.RawArray(data=np.array(x), info=info)

        mapping = {1: 'right_hand', 2: 'rest'}
        annot_from_events = mne.annotations_from_events(
            events=events, event_desc=mapping, sfreq=raw.info['sfreq'])
        raw.set_annotations(annot_from_events)

        return {"session_0": {"run_0": raw}}

    def data_path(
        self, path=None, force_update=False, update_path=None, verbose=None
    ):
        """Download the data from all the group"""
        path = '/Users/marieconstance.corsi/Documents/GitHub/Fenicotteri-equilibristi/Database/0_BCI/Classification/3_Dataset-netbci-meg-sess4-DK/MEG_DK.mat'
        return [path]

#%% parameters to be applied

dataset_MEG_DK = MEG_DK_Dataset()
subjects = dataset_MEG_DK.subject_list
events = ["right_hand", "rest"]

freqbands= {'theta': [5, 7],
            'alpha': [8,12],
            'beta': [13, 30],
            'paper': [3, 40]}

kk_components = 8
nbSplit = 50
# parameters for the default classifier & cross-validation
svm = GridSearchCV(SVC(), {"kernel": ("linear", "rbf"), "C": [0.1, 1, 10]}, cv=3)
csp = CSP_MNE(n_components=kk_components, reg=None, log=True, norm_trace=False)  # average power
cv = ShuffleSplit(nbSplit, test_size=0.2, random_state=21)

opt_atm_param_edge=pd.read_csv(
    path_csv_root + "/SVM/df_optcfg_SVM_Classification-allnode-2class-right_hand-rest-" + "n_csp_cmp-" + str(
        kk_components) + "-freq-" + str(3) + '-' + str(40) + '-nbSplit' + str(nbSplit) + ".csv")

#%%
for subject in subjects:

    for f in freqbands:
        fmin = freqbands[f][0]
        fmax = freqbands[f][1]

        results = pd.DataFrame()

        paradigm_meg = MotorImagery(
            events=events, n_classes=len(events), fmin=fmin, fmax=fmax
        )
        ep_meg, labels, meta = paradigm_meg.get_data(
            dataset=dataset_MEG_DK, subjects=[subject], return_epochs=True
        )
        epochs_data=ep_meg.get_data()
        nb_ROIs=np.shape(epochs_data)[1]
        nb_trials=np.shape(epochs_data)[0]

        class_balance = np.mean(labels == labels[0])
        class_balance = max(class_balance, 1. - class_balance)

        # get individual optimal parameters to make the process faster
        kk_zthresh = opt_atm_param_edge['zthresh'][subject]
        kk_val_duration = opt_atm_param_edge['val_duration'][subject]


        #%% ATM + SVM
        Nep_meg, labels, meta = paradigm_meg.get_data(
            dataset=dataset_MEG_DK, subjects=[subject], return_epochs=False
        )
        temp = np.transpose(Nep_meg, (1, 0, 2))
        temp_nc = np.reshape(temp, (np.shape(temp)[0], np.shape(temp)[1] * np.shape(temp)[2]))
        zscored_data = zscore(temp_nc, axis=1)
        # epoching here before computing the avalanches
        temp_zscored_data_ep = np.reshape(zscored_data, (np.shape(temp)[0], np.shape(temp)[1], np.shape(temp)[2]))
        zscored_data_ep = np.transpose(temp_zscored_data_ep, (1, 0, 2))

        ATM = np.empty((nb_trials, nb_ROIs, nb_ROIs))
        for kk_trial in range(nb_trials):
            list_avalanches, min_size_avalanches, max_size_avalanches, list_avalanches_bysize, temp_ATM = find_avalanches(
                zscored_data_ep[kk_trial, :, :], thresh=kk_zthresh, val_duration=kk_val_duration)
            # ATM: nb_trials x nb_ROIs x nb_ROIs matrix
            ATM[kk_trial, :, :] = temp_ATM

        clf_2 = Pipeline([('SVM', svm)])
        reshape_ATM = np.reshape(ATM, (np.shape(ATM)[0], np.shape(ATM)[1] * np.shape(ATM)[2]))
        score_ATM_SVM = cross_val_score(clf_2, reshape_ATM, labels, cv=cv, n_jobs=None)
        print("Classification accuracy ATM+SVM: %f / Chance level: %f" % (np.mean(score_ATM_SVM),
                                                                          class_balance))


        #%% concatenate results in a single dataframe
        pd_ATM_SVM = pd.DataFrame(score_ATM_SVM, columns=["ATM+SVM"])

        results = pd_ATM_SVM

        c_scores = score_ATM_SVM
        ppl = ["ATM+SVM"]*len(score_ATM_SVM)
        c_subject = [subject]*len(ppl)

        split = np.arange(nbSplit)
        zthresh = [kk_zthresh]*len(ppl)
        val_duration = [kk_val_duration] * len(ppl)
        n_csp_comp = [kk_components]*len(ppl)
        freq = [str(fmin)+'-'+str(fmax)]*len(ppl)

        data2=np.transpose(np.vstack((c_scores,ppl, split, n_csp_comp, zthresh, val_duration, freq, c_subject)))
        results=pd.DataFrame(data2,columns=["score","pipeline","split",
                                            "n_csp_comp", "zthresh", "val_duration", "freq",
                                            "subject"])
        results.to_csv(
            path_csv_root + "/SVM/IndivOpt_Comparison_SVM_Classification-allnode-2class-right_hand-rest-subject-" + str(subject) +  "n_csp_cmp-" + str(kk_components) + "zthresh-" + str(kk_zthresh) +
            "-freq-" + str(fmin) +'-'+ str(fmax) +'-nbSplit' + str(nbSplit) +
                 "_val_duration_" +str(kk_val_duration) + ".csv"
        )
        print(
            "saved " +
            path_csv_root + "/SVM/IndivOpt_Comparison_SVM_Classification-allnode-2class-right_hand-rest-subject-" + str(subject) +  "n_csp_cmp-" + str(kk_components) + "zthresh-" + str(kk_zthresh) +
            "-freq-" + str(fmin) +'-'+ str(fmax) +'-nbSplit' + str(nbSplit) +
                 "_val_duration_" +str(kk_val_duration) + ".csv"
        )

#%% load results to plot freq band effect
res2plot=pd.DataFrame()
res2plot_median = pd.DataFrame()

import seaborn as sns
for subject in subjects:

    for f in freqbands:
        fmin = freqbands[f][0]
        fmax = freqbands[f][1]

        # get individual optimal parameters to make the process faster
        kk_zthresh = opt_atm_param_edge['zthresh'][subject]
        kk_val_duration = opt_atm_param_edge['val_duration'][subject]

        temp_res=pd.read_csv(
            path_csv_root + "/SVM/IndivOpt_Comparison_SVM_Classification-allnode-2class-right_hand-rest-subject-" + str(subject) +  "n_csp_cmp-" + str(kk_components) + "zthresh-" + str(kk_zthresh) +
            "-freq-" + str(fmin) +'-'+ str(fmax) +'-nbSplit' + str(nbSplit) +
                 "_val_duration_" +str(kk_val_duration) + ".csv"
        )
        temp_median=temp_res.head(1)
        score_median=temp_res["score"].median()
        temp_median["score"]=score_median
        res2plot=pd.concat((temp_res, res2plot))
        res2plot_median = pd.concat((temp_median, res2plot_median))


#%%
import matplotlib.pyplot as plt
plt.style.use('default')
plt.style.use('seaborn-colorblind')
list_order=['5-7', '8-12', '13-30', '3-40']

path_figures_root = "/Users/marieconstance.corsi/Documents/GitHub/Fenicotteri-equilibristi/Figures/Classification/"
ax = sns.boxplot(data=res2plot_median,
                y="score",
                x="freq",
                 palette='viridis',
                 order=list_order),
                # whis=np.inf)
ax = sns.swarmplot(data=res2plot_median,
                y="score",
                x="freq", color='.2',
                order=list_order)
sns.despine()
plt.savefig(path_figures_root + "FreqInfluence_Rebuttal_iScience_SVM_Classification_edges_nodes-2class-nbSplits"+str(nbSplit)+"_medianGroup_MEG.png", dpi=300)

#%%
res_theta=res2plot_median[res2plot_median["freq"]=='5-7']
median_theta=res_theta["score"].median()
std_theta=res_theta["score"].std()

res_alpha=res2plot_median[res2plot_median["freq"]=='8-12']
median_alpha=res_alpha["score"].median()
std_alpha=res_alpha["score"].std()

res_beta=res2plot_median[res2plot_median["freq"]=='13-30']
median_beta=res_beta["score"].median()
std_beta=res_beta["score"].std()

res_broad=res2plot_median[res2plot_median["freq"]=='3-40']
median_broad=res_broad["score"].median()
std_broad=res_broad["score"].std()

#%% plot a random ATM matrix with ROIs labels
ROI_DK_list = [ 'bankssts L' , 'bankssts R' , 'caudalanteriorcingulate L' , 'caudalanteriorcingulate R' ,
                'caudalmiddlefrontal L' , 'caudalmiddlefrontal R' , 'cuneus L' , 'cuneus R' , 'entorhinal L' ,
                'entorhinal R' , 'frontalpole L' , 'frontalpole R' , 'fusiform L' , 'fusiform R' ,
                'inferiorparietal L' , 'inferiorparietal R' , 'inferiortemporal L' , 'inferiortemporal R' , 'insula L' ,
                'insula R' , 'isthmuscingulate L' , 'isthmuscingulate R' , 'lateraloccipital L' , 'lateraloccipital R' ,
                'lateralorbitofrontal L' , 'lateralorbitofrontal R' , 'lingual L' , 'lingual R' ,
                'medialorbitofrontal L' , 'medialorbitofrontal R' , 'middletemporal L' , 'middletemporal R' ,
                'paracentral L' , 'paracentral R' , 'parahippocampal L' , 'parahippocampal R' , 'parsopercularis L' ,
                'parsopercularis R' , 'parsorbitalis L' , 'parsorbitalis R' , 'parstriangularis L' ,
                'parstriangularis R' , 'pericalcarine L' , 'pericalcarine R' , 'postcentral L' , 'postcentral R' ,
                'posteriorcingulate L' , 'posteriorcingulate R' , 'precentral L' , 'precentral R' , 'precuneus L' ,
                'precuneus R' , 'rostralanteriorcingulate L' , 'rostralanteriorcingulate R' , 'rostralmiddlefrontal L' ,
                'rostralmiddlefrontal R' , 'superiorfrontal L' , 'superiorfrontal R' , 'superiorparietal L' ,
                'superiorparietal R' , 'superiortemporal L' , 'superiortemporal R' , 'supramarginal L' ,
                'supramarginal R' , 'temporalpole L' , 'temporalpole R' , 'transversetemporal L' ,
                'transversetemporal R' ]

#df_rand_atm=pd.DataFrame(ATM[0,:,:], columns=ROI_DK_list, index=ROI_DK_list)

nb_ROIS_DK = len(ROI_DK_list)
random_DK = np.random.random((nb_ROIS_DK,nb_ROIS_DK))
df_rand_DK= pd.DataFrame(random_DK, columns=ROI_DK_list, index=ROI_DK_list)

x,y = 18,18
plt.figure(figsize=(x,y))
ax=sns.heatmap(df_rand_DK, annot=False, square=True, cbar=False, xticklabels=True, yticklabels=True, cmap='YlGnBu')
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=11)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, fontsize=11, rotation_mode='anchor', ha='right')
ax.figure.savefig(path_figures_root + "RandATM_Matrix_ROIs_DK.png", dpi=300, transparent=True, bbox_inches='tight')



ROI_Destrieux_list = ['G_Ins_lg_and_S_Ment_ins L','G_Ins_lg_and_S_Ment_ins R','G_and_S_Mingul-Ant L','G_and_S_Mingul-Ant R','G_and_S_Mingul-Mid-Ant L','G_and_S_Mingul-Mid-Ant R','G_and_S_Mingul-Mid-Post L','G_and_S_Mingul-Mid-Post R','G_and_S_frontomargin L','G_and_S_frontomargin R','G_and_S_occipital_inf L','G_and_S_occipital_inf R','G_and_S_paracentral L','G_and_S_paracentral R','G_and_S_subcentral L','G_and_S_subcentral R','G_and_S_transv_frontopol L','G_and_S_transv_frontopol R','G_Mingul-Post-dorsal L','G_Mingul-Post-dorsal R','G_Mingul-Post-ventral L','G_Mingul-Post-ventral R','G_Muneus L','G_Muneus R','G_front_inf-Opercular L','G_front_inf-Opercular R','G_front_inf-Orbital L','G_front_inf-Orbital R','G_front_inf-Triangul L','G_front_inf-Triangul R','G_front_middle L','G_front_middle R','G_front_sup L','G_front_sup R','G_insular_short L','G_insular_short R','G_oc-temp_lat-fusifor L','G_oc-temp_lat-fusifor R','G_oc-temp_med-Lingual L','G_oc-temp_med-Lingual R','G_oc-temp_med-Parahip L','G_oc-temp_med-Parahip R','G_occipital_middle L','G_occipital_middle R','G_occipital_sup L','G_occipital_sup R','G_orbital L','G_orbital R','G_pariet_inf-Angular L','G_pariet_inf-Angular R','G_pariet_inf-Supramar L','G_pariet_inf-Supramar R','G_parietal_sup L','G_parietal_sup R','G_postcentral L','G_postcentral R','G_precentral L','G_precentral R','G_precuneus L','G_precuneus R','G_rectus L','G_rectus R','G_subcallosal L','G_subcallosal R','G_temp_sup-G_T_transv L','G_temp_sup-G_T_transv R','G_temp_sup-Lateral L','G_temp_sup-Lateral R','G_temp_sup-Plan_polar L','G_temp_sup-Plan_polar R','G_temp_sup-Plan_tempo L','G_temp_sup-Plan_tempo R','G_temporal_inf L','G_temporal_inf R','G_temporal_middle L','G_temporal_middle R','Lat_Fis-ant-Horizont L','Lat_Fis-ant-Horizont R','Lat_Fis-ant-Vertical L','Lat_Fis-ant-Vertical R','Lat_Fis-post L','Lat_Fis-post R','Pole_occipital L','Pole_occipital R','Pole_temporal L','Pole_temporal R','S_Malcarine L','S_Malcarine R','S_Mentral L','S_Mentral R','S_Mingul-Marginalis L','S_Mingul-Marginalis R','S_Mircular_insula_ant L','S_Mircular_insula_ant R','S_Mircular_insula_inf L','S_Mircular_insula_inf R','S_Mircular_insula_sup L','S_Mircular_insula_sup R','S_Mollat_transv_ant L','S_Mollat_transv_ant R','S_Mollat_transv_post L','S_Mollat_transv_post R','S_front_inf L','S_front_inf R','S_front_middle L','S_front_middle R','S_front_sup L','S_front_sup R','S_interm_prim-Jensen L','S_interm_prim-Jensen R','S_intrapariet_and_P_trans L','S_intrapariet_and_P_trans R','S_oc-temp_lat L','S_oc-temp_lat R','S_oc-temp_med_and_Lingual L','S_oc-temp_med_and_Lingual R','S_oc_middle_and_Lunatus L','S_oc_middle_and_Lunatus R','S_oc_sup_and_transversal L','S_oc_sup_and_transversal R','S_occipital_ant L','S_occipital_ant R','S_orbital-H_Shaped L','S_orbital-H_Shaped R','S_orbital_lateral L','S_orbital_lateral R','S_orbital_med-olfact L','S_orbital_med-olfact R','S_parieto_occipital L','S_parieto_occipital R','S_pericallosal L','S_pericallosal R','S_postcentral L','S_postcentral R','S_precentral-inf-part L','S_precentral-inf-part R','S_precentral-sup-part L','S_precentral-sup-part R','S_suborbital L','S_suborbital R','S_subparietal L','S_subparietal R','S_temporal_inf L','S_temporal_inf R','S_temporal_sup L','S_temporal_sup R','S_temporal_transverse L','S_temporal_transverse R']
nb_ROIS_Destrieux = len(ROI_Destrieux_list)
random_Destrieux = np.random.random((nb_ROIS_Destrieux,nb_ROIS_Destrieux))
df_rand_atm_Destrieux=pd.DataFrame(random_Destrieux, columns=ROI_Destrieux_list, index=ROI_Destrieux_list)

x,y = 17,17
plt.figure(figsize=(x,y))
ax=sns.heatmap(df_rand_atm_Destrieux, annot=False, square=True, cbar=False, xticklabels=True, yticklabels=True, cmap='YlGnBu')
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=6)
ax.set_xticklabels(ax.get_xticklabels(), rotation=60, fontsize=6, rotation_mode='anchor', ha='right')
#plt.show()
ax.figure.savefig(path_figures_root + "RandATM_Matrix_ROIs_Destrieux.png", dpi=300, transparent=True, bbox_inches='tight')
