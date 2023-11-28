import mne
import mat73
import numpy as np

from moabb.datasets.base import BaseDataset

#%% concat all - trials = pz
class Epilepsy_MEG_AAL_Dataset_concat(BaseDataset):
    """
    Dataset from the Epilepsy datasest, source space, MEG, AAL - trials = pz, 1 concatenated recording per subject
    """

    def __init__(self):
        super().__init__(
            subjects=[0],
            sessions_per_subject=1, # for the moment...
            events={"EP1": 1, "HC": 0},
            code="Epilepsy_MEG_AAL_Dataset_concat",
            interval=[0, 100864/256], # arbitrary, all data from RS recordings
            paradigm="imagery",
            doi="", # not available yet...
        )

    def _get_single_subject_data(self, subject):
        """return data for a single subject"""
        file_path = self.data_path(subject)

        temp = mat73.loadmat(file_path[0]) # huge files...
        x = temp['Data_moabb']#[subject] # 2D: nb rois x (nb_trials x nb_samples)
        fs = temp["fs"]
        ch_names = temp['labels_AAL1'] #list(np.transpose(temp['labels_rnd_AAL'])[0]) # actually, random roi names here...
        events=temp["Events_moabb"]#[subject] #1: EP1, 0: HC

        ch_types = ["eeg" for i in range(np.shape(ch_names)[0])]
        info = mne.create_info(ch_names, fs, ch_types)
        raw = mne.io.RawArray(data=np.array(x), info=info)

        mapping = {1: 'EP1', 0: 'HC'}
        annot_from_events = mne.annotations_from_events(
            events=events, event_desc=mapping, sfreq=raw.info['sfreq'])
        raw.set_annotations(annot_from_events)

        return {"session_0": {"run_0": raw}}

    def data_path(
        self, path=None, force_update=False, update_path=None, verbose=None
    ):
        """Download the data from all the group"""
        path = '/Users/marieconstance.corsi/Documents/GitHub/Fenicotteri-equilibristi/Database/1_Clinical/Epilepsy_GMD/Data_Epi_MEG_4Classif_concat_NoTrials.mat'

        return [path]
#%% 3s-trials -test
class Epilepsy_MEG_AAL_Dataset_3s_draft(BaseDataset):
    """
    Dataset from the Epilepsy datasest, source space, MEG, AAL - trials 3s
    """

    def __init__(self):
        super().__init__(
            subjects=[0],
            sessions_per_subject=64, # one per pz/subject
            events={"EP1": 1, "HC": 0},
            code="Epilepsy_MEG_AAL_Dataset_3s_draft",
            interval=[0, 3], # arbitrary trials from RS recordings
            paradigm="imagery",
            doi="", # not available yet...
        )

    def _get_single_subject_data(self, subject):
        """return data for a single subject"""
        file_path = self.data_path()
        temp = mat73.loadmat(file_path[0])  # huge files...

        out = {}
        for sess_ind in range(self.n_sessions):
            sess_key = "session_{}".format(sess_ind)
            out[sess_key] = {}
            run_ind=0
            run_key = "run_{}".format(run_ind)
            x = temp['Data_moabb'][sess_ind]  # 2D: nb rois x (nb_trials x nb_samples)
            fs = temp["fs"]
            ch_names = temp['labels_rnd_AAL']  # actually, random roi names here...
            events = temp["Events_moabb"][sess_ind]  # 1: EP1, 0: HC

            ch_types = ["eeg" for i in range(np.shape(ch_names)[0])]
            info = mne.create_info(ch_names, fs, ch_types)
            raw = mne.io.RawArray(data=np.array(x), info=info)

            mapping = {1: 'EP1', 0: 'HC'}
            annot_from_events = mne.annotations_from_events(
                events=events, event_desc=mapping, sfreq=raw.info['sfreq'])
            raw.set_annotations(annot_from_events)
            out[sess_key][run_key] = raw

        return out


        sessions = {}
        for sess in range(self.n_sessions)+1: #actually sessions = pz

            x = temp['Data_moabb'][sess] # 2D: nb rois x (nb_trials x nb_samples)
            fs = temp["fs"]
            ch_names = temp['labels_rnd_AAL'] # actually, random roi names here...
            events=temp["Events_moabb"][sess] #1: EP1, 0: HC

            ch_types = ["eeg" for i in range(np.shape(ch_names)[0])]
            info = mne.create_info(ch_names, fs, ch_types)
            raw = mne.io.RawArray(data=np.array(x), info=info)

            mapping = {1: 'EP1', 0: 'HC'}
            annot_from_events = mne.annotations_from_events(
                events=events, event_desc=mapping, sfreq=raw.info['sfreq'])
            raw.set_annotations(annot_from_events)

            sessions["session_"+str(sess)] = {}
            sessions["session_"+str(sess)]["run_1"] = raw

        return sessions


       # return {"session_0": {"run_0": raw}}

    def data_path(
        self, path=None, force_update=False, update_path=None, verbose=None
    ):
        """Download the data from all the group"""
        path = '/Users/marieconstance.corsi/Documents/GitHub/Fenicotteri-equilibristi/Database/1_Clinical/Epilepsy_GMD/Data_HC_EP1_MEG_DK_4Classif_3s_draft.mat'

        return [path]

#%% 3s-trials
class Epilepsy_MEG_AAL_Dataset_3s(BaseDataset):
    """
    Dataset from the Epilepsy datasest, source space, MEG, AAL - trials 3s
    """

    def __init__(self):
        super().__init__(
            subjects=[0],
            sessions_per_subject=1, # for the moment...
            events={"EP1": 1, "HC": 0},
            code="Epilepsy_MEG_AAL_Dataset_3s",
            interval=[0, 3], # arbitrary trials from RS recordings
            paradigm="imagery",
            doi="", # not available yet...
        )

    def _get_single_subject_data(self, subject):
        """return data for a single subject"""
        file_path = self.data_path(subject)

        temp = mat73.loadmat(file_path[0]) # huge files...
        x = temp['Data_moabb'][subject] # 2D: nb rois x (nb_trials x nb_samples)
        fs = temp["fs"]
        ch_names = list(np.transpose(temp['labels_rnd_AAL'])[0]) # actually, random roi names here...
        events=temp["Events_moabb"][subject] #1: EP1, 0: HC

        ch_types = ["eeg" for i in range(np.shape(ch_names)[0])]
        info = mne.create_info(ch_names, fs, ch_types)
        raw = mne.io.RawArray(data=np.array(x), info=info)

        mapping = {1: 'EP1', 0: 'HC'}
        annot_from_events = mne.annotations_from_events(
            events=events, event_desc=mapping, sfreq=raw.info['sfreq'])
        raw.set_annotations(annot_from_events)

        return {"session_0": {"run_0": raw}}

    def data_path(
        self, path=None, force_update=False, update_path=None, verbose=None
    ):
        """Download the data from all the group"""
        path = '/Users/marieconstance.corsi/Documents/GitHub/Fenicotteri-equilibristi/Database/1_Clinical/Epilepsy_GMD/Data_HC_EP1_MEG_DK_4Classif_3s.mat'

        return [path]


#%% 5s-trials
class Epilepsy_MEG_AAL_Dataset_5s(BaseDataset):
    """
    Dataset from the Epilepsy datasest, source space, MEG, AAL - trials 3s
    """

    def __init__(self):
        super().__init__(
            subjects=[0],
            sessions_per_subject=1, # for the moment...
            events={"EP1": 1, "HC": 0},
            code="Epilepsy_MEG_AAL_Dataset_3s",
            interval=[0, 5], # arbitrary trials from RS recordings
            paradigm="imagery",
            doi="", # not available yet...
        )

    def _get_single_subject_data(self, subject):
        """return data for a single subject"""
        file_path = self.data_path(subject)

        temp = mat73.loadmat(file_path[0]) # huge files...
        x = temp['Data_moabb'][subject] # 2D: nb rois x (nb_trials x nb_samples)
        fs = temp["fs"]
        ch_names = list(np.transpose(temp['labels_rnd_AAL'])[0]) # actually, random roi names here...
        events=temp["Events_moabb"][subject] #1: EP1, 0: HC

        ch_types = ["eeg" for i in range(np.shape(ch_names)[0])]
        info = mne.create_info(ch_names, fs, ch_types)
        raw = mne.io.RawArray(data=np.array(x), info=info)

        mapping = {1: 'EP1', 0: 'HC'}
        annot_from_events = mne.annotations_from_events(
            events=events, event_desc=mapping, sfreq=raw.info['sfreq'])
        raw.set_annotations(annot_from_events)

        return {"session_0": {"run_0": raw}}

    def data_path(
        self, path=None, force_update=False, update_path=None, verbose=None
    ):
        """Download the data from all the group"""
        path = '/Users/marieconstance.corsi/Documents/GitHub/Fenicotteri-equilibristi/Database/1_Clinical/Epilepsy_GMD/Data_HC_EP1_MEG_DK_4Classif_5s.mat'

        return [path]


#%% 8s-trials
class Epilepsy_MEG_AAL_Dataset_8s(BaseDataset):
    """
    Dataset from the Epilepsy datasest, source space, MEG, AAL - trials 3s
    """

    def __init__(self):
        super().__init__(
            subjects=[0],
            sessions_per_subject=1, # for the moment...
            events={"EP1": 1, "HC": 0},
            code="Epilepsy_MEG_AAL_Dataset_3s",
            interval=[0, 8], # arbitrary trials from RS recordings
            paradigm="imagery",
            doi="", # not available yet...
        )

    def _get_single_subject_data(self, subject):
        """return data for a single subject"""
        file_path = self.data_path(subject)

        temp = mat73.loadmat(file_path[0]) # huge files...
        x = temp['Data_moabb'][subject] # 2D: nb rois x (nb_trials x nb_samples)
        fs = temp["fs"]
        ch_names = list(np.transpose(temp['labels_rnd_AAL'])[0]) # actually, random roi names here...
        events=temp["Events_moabb"][subject] #1: EP1, 0: HC

        ch_types = ["eeg" for i in range(np.shape(ch_names)[0])]
        info = mne.create_info(ch_names, fs, ch_types)
        raw = mne.io.RawArray(data=np.array(x), info=info)

        mapping = {1: 'EP1', 0: 'HC'}
        annot_from_events = mne.annotations_from_events(
            events=events, event_desc=mapping, sfreq=raw.info['sfreq'])
        raw.set_annotations(annot_from_events)

        return {"session_0": {"run_0": raw}}

    def data_path(
        self, path=None, force_update=False, update_path=None, verbose=None
    ):
        """Download the data from all the group"""
        path = '/Users/marieconstance.corsi/Documents/GitHub/Fenicotteri-equilibristi/Database/1_Clinical/Epilepsy_GMD/Data_HC_EP1_MEG_DK_4Classif_8s.mat'

        return [path]

#%% 10s-trials
class Epilepsy_MEG_AAL_Dataset_10s(BaseDataset):
    """
    Dataset from the Epilepsy datasest, source space, MEG, AAL - trials 3s
    """

    def __init__(self):
        super().__init__(
            subjects=[0],
            sessions_per_subject=1, # for the moment...
            events={"EP1": 1, "HC": 0},
            code="Epilepsy_MEG_AAL_Dataset_3s",
            interval=[0, 10], # arbitrary trials from RS recordings
            paradigm="imagery",
            doi="", # not available yet...
        )

    def _get_single_subject_data(self, subject):
        """return data for a single subject"""
        file_path = self.data_path(subject)

        temp = mat73.loadmat(file_path[0]) # huge files...
        x = temp['Data_moabb'][subject] # 2D: nb rois x (nb_trials x nb_samples)
        fs = temp["fs"]
        ch_names = list(np.transpose(temp['labels_rnd_AAL'])[0]) # actually, random roi names here...
        events=temp["Events_moabb"][subject] #1: EP1, 0: HC

        ch_types = ["eeg" for i in range(np.shape(ch_names)[0])]
        info = mne.create_info(ch_names, fs, ch_types)
        raw = mne.io.RawArray(data=np.array(x), info=info)

        mapping = {1: 'EP1', 0: 'HC'}
        annot_from_events = mne.annotations_from_events(
            events=events, event_desc=mapping, sfreq=raw.info['sfreq'])
        raw.set_annotations(annot_from_events)

        return {"session_0": {"run_0": raw}}

    def data_path(
        self, path=None, force_update=False, update_path=None, verbose=None
    ):
        """Download the data from all the group"""
        path = '/Users/marieconstance.corsi/Documents/GitHub/Fenicotteri-equilibristi/Database/1_Clinical/Epilepsy_GMD/Data_HC_EP1_MEG_DK_4Classif_10s.mat'

        return [path]