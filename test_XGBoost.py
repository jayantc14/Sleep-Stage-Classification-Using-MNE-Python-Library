"""
By: Smayan Das
    Jayant Choudhary
"""



import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mne
from mne.time_frequency import psd_welch

from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import cohen_kappa_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn import model_selection

# The Sleep Physionet dataset is annotated using `8 labels <physionet_labels>`:
# Wake (W), Stage 1, Stage 2, Stage 3, Stage 4 corresponding to the range from
# light sleep to deep sleep, REM sleep (R) where REM is the abbreviation for
# Rapid Eye Movement sleep, movement (M), and Stage (?) for any none scored
# segment.
#
# We will work only with 5 stages: Wake (W), Stage 1, Stage 2, Stage 3/4, and
# REM sleep (R). To do so, we use the ``event_id`` parameter in
# :func:`mne.events_from_annotations` to select which events are we
# interested in and we associate an event identifier to each of them.

import warnings
warnings.filterwarnings('ignore')
#Setting the channel types : We have only used the EEG(fpz-pz) and EOG(eog_horizontal) channels
mapping = {'EOG horizontal': 'misc',
           'Resp oro-nasal': 'misc',
           'EMG submental': 'misc',
           'Temp rectal': 'misc',
           'Event marker': 'misc'}

# Mapping the Annotations to event_ids
annotation_desc_2_event_id = {'Sleep stage W': 1,
                              'Sleep stage 1': 2,
                              'Sleep stage 2': 3,
                              'Sleep stage 3': 4,
                              'Sleep stage 4': 4,
                              'Sleep stage R': 5}

# Creating a new event_id that unifies stages 3 and 4 according to AASM standards
event_id = {'Sleep stage W': 1,
            'Sleep stage 1': 2,
            'Sleep stage 2': 3,
            'Sleep stage 3/4': 4,
            'Sleep stage R': 5}

# Using Pandas, the filenames are extracted from which the data processing starts.
df = pd.read_csv("data_records_sc.csv")
df.columns=["file_name"]
rawlist=list(df.file_name)

items = len(rawlist)
print(items)

#Creating an empty list to store the training epochs data
epochs_data_train = []

for item in range(0,240,2):
    """
    The below operations read the edf PSG files, add the annotations via event markers to the raw edf
    """
    raw = mne.io.read_raw_edf(f"{rawlist[item]}")
    annot = mne.read_annotations(f"{rawlist[item + 1]}")
    raw.set_annotations(annot, emit_warning=False)
    raw.set_channel_types(mapping)
    # raw.resample(1)
    # raw.plot(duration=60, scalings='auto')
    # print(raw)
    # print(raw.info)
    # print(annot)
    events, _ = mne.events_from_annotations(
        raw, event_id=annotation_desc_2_event_id, chunk_duration=30.)

    # Create Epochs from the data based on the events found in the annotations
    tmax = 30. - 1. / raw.info['sfreq']  # tmax in included
    # Try Except method was used as some files were showing errors while getting parsed by MNE and we have ignored them for now.
    try:
        epochs = mne.Epochs(raw=raw, events=events,event_id=event_id, tmin=0., tmax=tmax, baseline=None)
    except:
        continue
    # It was observed that there were no bad annotations in the whole dataset so we commented out the following line
    # epochs.drop_bad()
    print("len(epochs)")
    #Adding the epoch to the list of training epochs
    epochs_data_train.append(epochs)
    print(len(epochs_data_train))

#This function concatenates all the training epochs as metadata without any explicit markers.
epochs_total_train = mne.concatenate_epochs(epochs_list=epochs_data_train)

# The above method for creating Training data is followed as it is for creating the Test data

#Creating an empty list to store the test epochs.

epochs_data_test = []

# The indices 2 and 30 can be randomized which we are yet to implement.
for item in range(240,300,2):
    raw = mne.io.read_raw_edf(f"{rawlist[item]}")
    annot = mne.read_annotations(f"{rawlist[item + 1]}")
    raw.set_annotations(annot, emit_warning=False)
    raw.set_channel_types(mapping)
    # raw.resample(1)
    # raw.plot(duration=60, scalings='auto')
    # print(raw)
    # print(raw.info)
    # print(annot)
    events, _ = mne.events_from_annotations(
        raw, event_id=annotation_desc_2_event_id, chunk_duration=30.)

    tmax = 30. - 1. / raw.info['sfreq']  # tmax in included

    try:
        epochs = mne.Epochs(raw=raw, events=events,event_id=event_id, tmin=0., tmax=tmax, baseline=None)
    except:
        continue

    # epochs.drop_bad()

    epochs_data_test.append(epochs)
    print(len(epochs_data_test))

#This function concatenates all the test epochs as metadata without any explicit markers.
epochs_total_test = mne.concatenate_epochs(epochs_list=epochs_data_test)
# print(epochs_total)
# print("\n\n\n\n\n")
# print(epochs_data)
# print(type(epochs_data))
# print(len(epochs_data))
# print(type(epochs_set))
# print(epochs_set)

def eeg_power_band(epochs):
    """EEG relative power band feature extraction.

    This function takes an ``mne.Epochs`` object and creates EEG features based
    on relative power in specific frequency bands that are compatible with
    scikit-learn.

    Parameters
    ----------
    epochs : Epochs
        The data.

    Returns
    -------
    X : numpy array of shape [n_samples, 5]
        Transformed data.
    """
    # specific frequency bands
    FREQ_BANDS = {"delta": [0.5, 4.5],
                  "theta": [4.5, 8.5],
                  "alpha": [8.5, 11.5],
                  "sigma": [11.5, 15.5],
                  "beta": [15.5, 30]}

    psds, freqs = psd_welch(epochs, picks='eeg', fmin=0.5, fmax=30.)
    # Normalize the PSDs
    psds /= np.sum(psds, axis=-1, keepdims=True)

    X = []
    for fmin, fmax in FREQ_BANDS.values():
        psds_band = psds[:, :, (freqs >= fmin) & (freqs < fmax)].mean(axis=-1)
        X.append(psds_band.reshape(len(psds), -1))

    return np.concatenate(X, axis=1)


# Multiclass Classifications using Function Transformer.

"""
Scikit-learn pipeline composes an estimator as a sequence of transforms
and a final estimator, while the FunctionTransformer converts a python
function in an estimator compatible object. In this manner a
scikit-learn estimator is created that takes :class:`mne.Epochs` using
`eeg_power_band` function.
"""

pipe = make_pipeline(FunctionTransformer(eeg_power_band, validate=False),
                     XGBClassifier())


y_train = epochs_total_train.events[:, 2]
pipe.fit(epochs_total_train, y_train)

y_pred_train = pipe.predict(epochs_total_train)
y_pred_test = pipe.predict(epochs_total_test)
y_test = epochs_total_test.events[:, 2]


acc_train = accuracy_score(y_train,y_pred_train)
acc_test = accuracy_score(y_test, y_pred_test)
print("Training Accuracy score: {}\n".format(acc_train))
print("Test Accuracy score: {}\n".format(acc_test))

kappa = cohen_kappa_score(y_test, y_pred_test)
print("Kohen Kappa Score: {}".format(kappa))

print(confusion_matrix(y_test, y_pred_test))

print(classification_report(y_test, y_pred_test, target_names=event_id.keys()))

# cv_score = model_selection.cross_val_score(pipe, epochs_total_train, y_train, cv=3)
#
# print(cv_score)


