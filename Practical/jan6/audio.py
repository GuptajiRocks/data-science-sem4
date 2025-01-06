import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

def visualize_audio(audio_path):
    y, sr = librosa.load(audio_path)

    plt.figure(figsize=(15, 10))

    plt.subplot(3, 2, 1)
    librosa.display.waveshow(y, sr=sr)
    plt.title("Waveform")

    X = librosa.stft(y)
    Xdb = librosa.amplitude_to_db(abs(X))
    plt.subplot(3, 2, 2)
    librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')
    plt.colorbar()
    plt.title("Spectrogram")

    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
    plt.subplot(3, 2, 3)
    librosa.display.specshow(log_mel_spectrogram, sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar()
    plt.title("Mel Spectrogram")

    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    plt.subplot(3, 2, 4)
    librosa.display.specshow(chroma, sr=sr, x_axis='time', y_axis='chroma')
    plt.colorbar()
    plt.title("Chroma Features")

    spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    frames = range(len(spectral_centroids))
    t = librosa.frames_to_time(frames)
    plt.subplot(3, 2, (5,6))
    librosa.display.waveshow(y, sr=sr, alpha=0.4)
    plt.plot(t, spectral_centroids, color='r')
    plt.title("Spectral Centroid")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    audio_file = "Practical\\jan6\\audio.wav"
    visualize_audio(audio_file)