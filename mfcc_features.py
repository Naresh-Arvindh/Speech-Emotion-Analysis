import numpy as np
import librosa

def get_audio_features(data, sr):
    mfcc = librosa.feature.mfcc(
        y=data,
        sr=sr,
        n_mfcc=13
    )
    delta_mfcc = librosa.feature.delta(mfcc)
    delta2_mfcc = librosa.feature.delta(mfcc, order=2)
    
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)
    
    delta_mean = np.mean(delta_mfcc, axis=1)
    delta_std = np.std(delta_mfcc, axis=1)
    delta2_mean = np.mean(delta2_mfcc, axis=1)
    delta2_std = np.std(delta2_mfcc, axis=1)
    
    features = np.hstack(
        (mfcc_mean,
         mfcc_std)
    )
    
    return features
