#%%
"""
==============================================================
Attempt to classify EEG data in the source space - neuronal avalanches vs classical approaches - classification on longer trials w/ SVM
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

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import SVC


import numpy as np

from scipy.stats import zscore

from moabb.paradigms import MotorImagery

import matplotlib.pyplot as plt
import ptitprince as pt


if os.path.basename(os.getcwd()) == "Fenicotteri-equilibristi":
    os.chdir("Database/0_BCI/Classification/")
if os.path.basename(os.getcwd()) == "py_viz":
    os.chdir("/Users/marieconstance.corsi/Documents/GitHub/Fenicotteri-equilibristi/Database/0_BCI/Classification/")
basedir = os.getcwd()

path_csv_root = os.getcwd() + '/1_Dataset-csv/NETBCI_EEG_DK_Sess4'
if not osp.exists(path_csv_root):
    os.mkdir(path_csv_root)
path_data_root = os.getcwd() + '/3_Dataset-netbci-EEG-sess4-DK'
if not osp.exists(path_data_root):
    os.mkdir(path_data_root)
path_data_root_chan = os.getcwd() + '/3_Dataset-netbci-eeg-sess4-DK'
#%% functions

def transprob(aval,nregions): # (t,r)
    mat = np.zeros((nregions, nregions))
    norm = np.sum(aval, axis=0)
    for t in range(len(aval) - 1):
        ini = np.where(aval[t] == 1)
        mat[ini] += aval[t + 1]
    mat[norm != 0] = mat[norm != 0] / norm[norm != 0][:, None]
    return mat

def Transprob(ZBIN,nregions, val_duration): # (t,r)
    mat = np.zeros((nregions, nregions))
    A = np.sum(ZBIN, axis=1)
    a = np.arange(len(ZBIN))
    idx = np.where(A != 0)[0]
    aout = np.split(a[idx], np.where(np.diff(idx) != 1)[0] + 1)
    #print(aout,np.shape(aout))
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
class EEG_DK_Dataset(BaseDataset):
    """
    Dataset from the NETBCI protocol, source space, EEG, DK, Sess4
    """

    def __init__(self):
        super().__init__(
            subjects=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19],
            sessions_per_subject=1, # for the moment...
            events={"right_hand": 1, "rest": 2},
            code="NETBCI_EEG_DK_Dataset",
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
        #subj_ID = temp['subject_IDs'][subject]
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
        path = '/Users/marieconstance.corsi/Documents/GitHub/Fenicotteri-equilibristi/Database/0_BCI/Classification/3_Dataset-netbci-EEG-sess4-DK/EEG_DK.mat'
        return [path]

#%%

from sklearn.pipeline import Pipeline
from sklearn.model_selection import ShuffleSplit, cross_val_score

dataset_EEG_DK = EEG_DK_Dataset()
subjects = dataset_EEG_DK.subject_list[:1] # too long otherwise...
events = ["right_hand", "rest"]

freqbands= {'paper': [3, 40]}

# parameters to compute the avalanches...
opt_zthresh = [3]
opt_val_duration = [3]

kk_components = 8

#%% perform the classification
for subject in subjects:

    for f in freqbands:
        fmin = freqbands[f][0]
        fmax = freqbands[f][1]

        results = pd.DataFrame()

        paradigm_EEG = MotorImagery(
            events=events, n_classes=len(events), fmin=fmin, fmax=fmax
        )
        ep_EEG, labels, meta = paradigm_EEG.get_data(
            dataset=dataset_EEG_DK, subjects=[subject], return_epochs=True
        )
        epochs_data=ep_EEG.get_data()
        nb_ROIs=np.shape(epochs_data)[1]
        nb_trials=np.shape(epochs_data)[0]

        class_balance = np.mean(labels == labels[0])
        class_balance = max(class_balance, 1. - class_balance)

        # Preparation of the next steps - Computing CSP with MNE & applying it to time series to obtain new epochs
        csp_mne = CSP_MNE(n_components=kk_components, transform_into='csp_space').fit(epochs_data, labels)
        epochs_data_csp_m = csp_mne.transform(epochs_data)
        info = create_info([f'CSP{i}' for i in range(kk_components)], sfreq=ep_EEG.info['sfreq'], ch_types='eeg')
        ep_csp_m = EpochsArray(epochs_data_csp_m, info, ep_EEG.events, event_id=ep_EEG.event_id)

        # parameters for the default classifier & cross-validation
        svm = GridSearchCV(SVC(), {"kernel": ("linear", "rbf"), "C": [0.1, 1, 10]}, cv=3)
        csp = CSP_MNE(n_components=kk_components, reg=None, log=True, norm_trace=False) # average power
        nbSplit=50#75#50#10
        cv = ShuffleSplit(nbSplit,test_size=0.2, random_state=21)

        #%% CSP + SVM: classical
        clf_0 = Pipeline([('CSP', csp), ('SVM', svm)])
        score_CSP_SVM = cross_val_score(clf_0, epochs_data, labels, cv=cv, n_jobs=None)
        print("Classification accuracy CSP+SVM: %f / Chance level: %f" % (np.mean(score_CSP_SVM),
                                                                  class_balance))

        for kk_zthresh in opt_zthresh:
            for kk_val_duration in opt_val_duration:
                #%% ATM + SVM
                Nep_EEG, labels, meta = paradigm_EEG.get_data(
                    dataset=dataset_EEG_DK, subjects=[subject], return_epochs=False
                )
                temp = np.transpose(Nep_EEG, (1, 0, 2))
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
                pd_CSP_SVM=pd.DataFrame(score_CSP_SVM, columns=["CSP+SVM"])
                pd_ATM_SVM = pd.DataFrame(score_ATM_SVM, columns=["ATM+SVM"])

                results = pd.concat([pd_CSP_SVM, pd_ATM_SVM,
                                     ],axis=1)

                c_scores = np.concatenate((score_CSP_SVM, score_ATM_SVM))
                ppl = np.concatenate((["CSP+SVM"]*len(score_CSP_SVM),
                                      ["ATM+SVM"]*len(score_ATM_SVM)))
                c_subject = [subject]*len(ppl)

                split = np.concatenate((np.arange(nbSplit),
                                        np.arange(nbSplit)))
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
                         "_val_duration_" +str(kk_val_duration) + "_EEG.csv"
                )
                print(
                    "saved " +
                    path_csv_root + "/SVM/IndivOpt_Comparison_SVM_Classification-allnode-2class-right_hand-rest-subject-" + str(subject) +  "n_csp_cmp-" + str(kk_components) + "zthresh-" + str(kk_zthresh) +
                    "-freq-" + str(fmin) +'-'+ str(fmax) +'-nbSplit' + str(nbSplit) +
                         "_val_duration_" +str(kk_val_duration) + "_EEG.csv"
                )

#%% study Diff
import statsmodels.stats.multitest
opt_zthresh = [1.4, 1.6, 1.8, 2, 2.2, 2.4, 2.6, 2.8, 3]
opt_val_duration = [2, 3, 4, 5, 6, 7, 8]
kk_components = 8
freqbands= {'paper': [3, 40]}

nbSplit=50
res_csv=pd.DataFrame()
res_diff_csv=pd.DataFrame()
res_csv_global = pd.DataFrame()
results_ttest=dict()
nb_split=list(range(0,nbSplit))

for f in freqbands:
    fmin = freqbands[f][0]
    fmax = freqbands[f][1]
    for kk_val_duration in opt_val_duration:
        for kk_zthresh in opt_zthresh:
            res_csv = pd.DataFrame()
            res_diff_csv = pd.DataFrame()
            x_Diff=[]
            vect_sign=[]
            pval_wilcoxon=[]
            pval_ttest=[]
            Diff = dict()
            Diff_median = []
            Diff_mean = []
            for subj in tqdm(subjects, desc="subject"):
                print(str(subj))
                temp = pd.read_csv(
                    path_csv_root + "/SVM/IndivOpt_Comparison_SVM_Classification-allnode-2class-right_hand-rest-subject-" + str(subj) +  "n_csp_cmp-" + str(kk_components) + "zthresh-" + str(kk_zthresh) +
                    "-freq-" + str(fmin) +'-'+ str(fmax) +'-nbSplit' + str(nbSplit) +
                         "_val_duration_" +str(kk_val_duration) + "_EEG.csv")

                sc_pipeline=temp.loc[temp['pipeline'] == 'ATM+SVM', 'score']
                Baseline=temp.loc[temp['pipeline'] == 'CSP+SVM', 'score']

                Diff[subj]  = list(Baseline.values - sc_pipeline.values)

                if min(Diff[subj]) == 0 :
                    x_Diff = np.sign(max(Diff[subj]))
                elif max(Diff[subj]) == 0:
                    x_Diff = np.sign(min(Diff[subj]))
                else:
                    x_Diff = np.sign(min(Diff[subj]))*np.sign(max(Diff[subj]))
                vect_sign.append(x_Diff)
                Diff_median.append(np.median(Diff[subj]))
                Diff_mean.append(np.mean(Diff[subj]))
                results_wilcoxon = scipy.stats.wilcoxon(Baseline.values,sc_pipeline.values)
                results_ttest = scipy.stats.ttest_rel(Baseline.values,sc_pipeline.values)
                pval_wilcoxon.append(results_wilcoxon.pvalue)
                pval_ttest.append(results_ttest.pvalue)

            [reject_wilcoxon, pval_wilcoxon_corrected, alphacSidak, alphacBonf] = statsmodels.stats.multitest.multipletests(pvals=pval_wilcoxon,
                                                     alpha=0.05, method='fdr_bh')#method='bonferroni')
            [reject_ttest, pval_ttest_corrected, alphacSidak, alphacBonf] = statsmodels.stats.multitest.multipletests(pvals=pval_ttest,
                                                     alpha=0.05, method='fdr_bh')#method='bonferroni')


            filename = path_csv_root + "/SVM/IndivOpt_ComparisonDiff_SVM_Classification-allnode-2class-right_hand-rest-" + "n_csp_cmp-" + str(kk_components) + "zthresh-" + str(kk_zthresh) + "-freq-" + str(fmin) + '-' + str(fmax) + '-nbSplit' + str(nbSplit) + "_val_duration_" + str(kk_val_duration) + "_EEG"

            with open(filename + '.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
                    pickle.dump([Diff, Diff_median, Diff_mean, vect_sign, pval_wilcoxon, pval_wilcoxon_corrected, pval_ttest, pval_ttest_corrected, nbSplit], f)


## TODO: cf notebook for the associated plots