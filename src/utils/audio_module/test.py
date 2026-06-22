import os
from pathlib import Path
import numpy as np
from scipy.io import wavfile

id = 'kthjazz_gCH_sFM_sngl_d01_007'
# id2='kthjazz_gCH_sFM_sngl_d01_008'

sr, wav = wavfile.read(f'./data/test_data/{id}.wav')
feature = np.load(f'./data/jukebox_features_sliced/{id}_1.npy')
# sr2, wav2 = wavfile.read(f'./data/test_data/{id2}.wav')
feature2 = np.load(f'./data/jukebox_features_sliced/{id}_333.npy')

test=1