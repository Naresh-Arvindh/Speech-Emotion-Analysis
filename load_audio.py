import librosa

def load_and_clean_audio(file_path):

    y, sr = librosa.load(file_path, sr=None, res_type='kaiser_fast')
    
    y_trimmed, _ = librosa.effects.trim(y, top_db=20)
    
    return y_trimmed, sr
