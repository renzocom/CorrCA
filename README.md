## CorrCA: Correlated Component Analysis

This repository contains the implementation of Correlated Component Analysis (CorrCA) based on the [original Matlab code](https://www.parralab.org/corrca/) from Parra's lab.

## Usage
Below is a simple example script demonstrating how to compute CorrCA on EEG evoked data.
'''
import numpy as np
from corrca import CorrCA

# Load your preprocessed EEG data as a NumPy array
# epochs: shape (n_epochs, n_channels, n_times)
# times: shape (n_times,)
epochs = np.load('path/to/your/epochs.npy')
times = np.load('path/to/your/times.npy')

# Define CorrCA parameters
params = {'baseline_window': (-0.3, -0.05), 'response_window': (0., 0.6), 'gamma': 0, 'K': 60, 'stats': True, 'n_surrogates': 500, 'alpha': 0.01}

# Perform CorrCA
W, ISC, A, Y, Yfull, ISC_thr = CorrCA.calc_corrca(epochs, times, **params)
'''
