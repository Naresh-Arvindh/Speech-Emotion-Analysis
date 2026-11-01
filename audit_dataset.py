import os
from collections import Counter
import kagglehub

KAGGLE_DATASET_PATH = kagglehub.dataset_download("uwrfkaggler/ravdess-emotional-speech-audio")

emotion_map = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

emotion_counts = Counter()
invalid_files = []
total_wav_count = 0

for root, _, files in os.walk(KAGGLE_DATASET_PATH):
    for file in files:
        if file.endswith(".wav"):
            total_wav_count += 1
            

            filename_without_ext = os.path.splitext(file)[0]
            parts = filename_without_ext.split("-")

            if len(parts) != 7:
                invalid_files.append(file)
                continue

            emotion_code = parts[2]

            if emotion_code not in emotion_map:
                invalid_files.append(file)
                continue

            emotion_counts[emotion_map[emotion_code]] += 1

print("\nDATASET AUDIT REPORT: ")
print("\nTotal WAV files found:", total_wav_count)
print("\nEmotion counts:")

for emotion, count in sorted(emotion_counts.items()):
    print(f"{emotion}: {count}")

print("\nInvalid filenames:", len(invalid_files))
print("Total files with valid emotion labels:", sum(emotion_counts.values()))
