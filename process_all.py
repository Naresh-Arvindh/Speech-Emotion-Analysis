import os
import numpy as np
import kagglehub
from load_audio import load_and_clean_audio
from mfcc_features import get_audio_features

def run_preprocessing_pipeline():
    X = []
    labels = []
    
    emotion_map = {
        "01": "neutral", "02": "calm", "03": "happy", "04": "sad",
        "05": "angry", "06": "fearful", "07": "disgust", "08": "surprised"
    }
    
    Dynamic_dataSet_Path = kagglehub.dataset_download("uwrfkaggler/ravdess-emotional-speech-audio")
    
    for root, _, files in os.walk(Dynamic_dataSet_Path):
        for file in files:
            if file.endswith(".wav"):
                parts = file.split("-")
                if len(parts) >= 3:
                    emotion_code = parts[2] 
                    if emotion_code in emotion_map:
                        full_path = os.path.join(root, file)
                        
                        data, sr = load_and_clean_audio(full_path)
                        features = get_audio_features(data, sr)
                        
                        X.append(features)
                        labels.append(emotion_map[emotion_code])
                        
    X = np.array(X)
    y = np.array(labels)
    
    np.save("X_data.npy", X)
    np.save("y_data.npy", y)
    print(f"Baseline extraction complete! Processed {len(X)} files.")

if __name__ == "__main__":
    run_preprocessing_pipeline()
