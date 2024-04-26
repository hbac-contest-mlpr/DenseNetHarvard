from scipy import signal
import spkit as sp
from sklearn.decomposition import FastICA
import numpy as np
from rich import print
import pandas as pd


# def butterworth(eeg_subsample, frequency=200, n_components=19):

#     return Xf


# def spkit(eeg_subsample, frequency=200, n_components=19):
#     Xf = sp.filter_X(
#         eeg_subsample, band=[0.70], btype="highpass", fs=frequency, verbose=1
#     )
#     return Xf


def ica(Xf, n_components=19):
    ica = FastICA(n_components=n_components, random_state=0, max_iter=1000, tol=1e-2)
    XR = ica.fit_transform(Xf)

    return Xf - XR


if __name__ == "__main__":

    np.random.seed(0)
    eeg_subsample = np.random.rand(5000, 19)
    frequency = 200
    n_components = 19
    
    df = pd.DataFrame(eeg_subsample)
    # print(df.head(10))
    # X = np.array([])
    x1 = np.expand_dims(eeg_subsample, 0)
    x2 = np.expand_dims(eeg_subsample, 0)
    x3 = np.vstack((x1, x2))
    print(x3.shape)

    # if the length is < 10000, we need to pad the data
    if len(eeg_subsample) < 10000:
        eeg_subsample = np.pad(eeg_subsample, ((0, 10000 - len(eeg_subsample)), (0, 0)), 'constant', constant_values=0)
    else:
        eeg_subsample = eeg_subsample[:10000]
    # similarily, if a column is missing, we need to add a column of zeros
    if eeg_subsample.shape[1] < 19:
        eeg_subsample = np.pad(eeg_subsample, ((0, 0), (0, 19 - eeg_subsample.shape[1])), 'constant', constant_values=0)
    else:
        eeg_subsample = eeg_subsample[:, :19]
    
    print(eeg_subsample.shape)

    outputs = np.ndarray((0, 19))

    # print("EEG Subsample: ", eeg_subsample)

    # filtered = butterworth(eeg_subsample.copy(), frequency, n_components)
    # spkit_filtered = spkit(eeg_subsample.copy(), frequency, n_components)

    # print("Butterworth: ", filtered)
    # print("SPKIT: ", spkit_filtered)

    # ica_butter = ica(filtered, n_components)
    # ica_spkit = ica(spkit_filtered, n_components)

    # print("ICA Butterworth: ", ica_butter)
    # print("ICA SPKIT: ", ica_spkit)
