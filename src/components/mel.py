import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os
# 1. Load the audio file
# sr=None preserves the native sampling rate
from pathlib import Path

folder_path = Path('dataset/Trail_Audio')
output_folder = "/content/drive/MyDrive/temp/Trail_Mel"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    print(f"Created directory: {output_folder}")
for file in folder_path.iterdir():
    # print(file.stem)
    file_name=file.stem
    # audio_path = 'your_file.wav'
    y, sr = librosa.load(file, sr=22050) 

    # 2. Compute the Mel Spectrogram
    # n_mels defines the "height" of your image (number of frequency bins)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)

    # 3. Convert to log scale (decibels)
    # This is crucial for neural networks to "see" the features clearly
    S_dB = librosa.power_to_db(S, ref=np.max)

    # 4. Visualize
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr, fmax=8000)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel-frequency spectrogram')
    plt.tight_layout()
    save_path = os.path.join(output_folder, f"{file_name}.jpg")
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()
    plt.close()