import numpy as np
from scipy.io import wavfile

def load_and_clean_audio(file_path):
    # This is exactly your original reading logic
    sr, data = wavfile.read(file_path)
    
    data = data.astype(float)
    
    if len(data.shape) > 1:
        data = data.mean(axis=1)
        
    data = data - np.mean(data)
    
    return data, sr
