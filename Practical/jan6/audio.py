import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

def visualize_audio(audio_path):
    y, sr = librosa.load(audio_path)

    plt.figure(figsize=(15, 10))

    X = librosa.stft(y)
    Xdb = librosa.amplitude_to_db(abs(X))
    librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')
    plt.colorbar()
    plt.title("Spectrogram")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    audio_file = "Practical\\jan6\\audio.wav"
    visualize_audio(audio_file)