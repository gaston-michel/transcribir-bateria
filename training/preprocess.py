from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift
import librosa
import matplotlib.pyplot as plt
import numpy as np
import os

import config
from filters import apply_lowpass_filter


def preprocess_wav_files(directory, sr=44100, clip_size=200, n_fft=1024, hop_length=512, classes=None):
    """
    Preprocesses WAV files from a directory, optionally filtering by classes.

    Args:
        directory (str): Path to the directory containing audio files.
        sr (int): Target sample rate for audio.
        clip_size (int): Clip duration in miliseconds.
        n_fft (int): FFT window size for spectrogram calculation.
        hop_length (int): Hop length for spectrogram calculation.
        classes (list, optional): List of class labels to filter files.

    Returns:
        tuple: A tuple containing the preprocessed spectrograms and labels.
    """
    spectrograms = []
    labels = []

    if classes is None:
        classes = os.listdir(directory)

    for i, label in enumerate(classes):
        print(f"Processing class {i+1}/{len(classes)}: {label}")
        process_label_directory(directory, label, spectrograms, labels, sr, clip_size, n_fft, hop_length)
    
    return np.array(spectrograms), np.array(labels)


def process_label_directory(directory, label, spectrograms, labels, sr, clip_size, n_fft, hop_length):
    """
    Processes all WAV files in a given label directory.

    Args:
        directory (str): Path to the base directory.
        label (str): The class label (subdirectory name).
        spectrograms (list): List to store spectrograms.
        labels (list): List to store labels.
        sr (int): Target sample rate for audio.
        clip_size (int): Clip duration in miliseconds.
        n_fft (int): FFT window size for spectrogram calculation.
        hop_length (int): Hop length for spectrogram calculation.
    """
    label_dir = os.path.join(directory, label)
    
    if not os.path.isdir(label_dir):
        print(f"Directory {label_dir} not found.")
        return
    
    for filename in os.listdir(label_dir):
        if filename.endswith(".wav"):
            process_wav_file(label_dir, filename, label, spectrograms, labels, sr, clip_size, n_fft, hop_length)


def process_wav_file(label_dir, filename, label, spectrograms, labels, sr, clip_size, n_fft, hop_length):
    """
    Processes a single WAV file, converts it to a spectrogram, and appends to lists.

    Args:
        label_dir (str): Path to the label directory.
        filename (str): The name of the WAV file.
        label (str): The class label.
        spectrograms (list): List to store spectrograms.
        labels (list): List to store labels.
        sr (int): Target sample rate for audio.
        clip_size (int): Clip duration in miliseconds.
        n_fft (int): FFT window size for spectrogram calculation.
        hop_length (int): Hop length for spectrogram calculation.
    """
    filepath = os.path.join(label_dir, filename)
    audio, _ = load_wav_file(filepath, sr)
    
    if len(audio) != sr * clip_size / 1000:
        print(f"Skipped {filename}. Audio length is not correct.")
        return
    
    if config.APPLY_LOW_FILTER:
        audio = apply_lowpass_filter(audio, cutoff_freq=1000)

    spectrogram = wav_to_spectrogram(audio, sr=sr, n_fft=n_fft, hop_length=hop_length)
    spectrograms.append(spectrogram)
    labels.append(label)

    if config.DATA_AUGMENTATION:
        augment = Compose([
            AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
            TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
            PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
            Shift(min_shift=-0.1, max_shift=0.5, p=0.5)
        ])
        augmented_audio = augment(audio, sample_rate=sr)
        augmented_spectrogram = wav_to_spectrogram(augmented_audio, sr=sr, n_fft=n_fft, hop_length=hop_length)
        spectrograms.append(augmented_spectrogram)
        labels.append(label)



def load_wav_file(filepath, target_sr=44100):
    """Loads a WAV file and resamples it to the target sample rate."""
    audio, sr = librosa.load(filepath, sr=target_sr)
    return audio, sr


def wav_to_spectrogram(audio, sr=44100, n_fft=1024, hop_length=512):
    """Converts a WAV audio to a Mel spectrogram."""
    spectrogram = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length
    )
    spectrogram_db = librosa.amplitude_to_db(abs(spectrogram))

    spectrogram_normalized = spectrogram_db / np.max(spectrogram_db)
    return spectrogram_normalized


def show_spectrogram_grid(spectrograms, labels, sr=44100, hop_length=256):
    """
    Displays one spectrogram for each unique label in a grid.

    Args:
        spectrograms (numpy.ndarray): Array of spectrograms.
        labels (numpy.ndarray): Array of corresponding labels.
        sr (int): Sample rate of the audio. Default is 22050.
        hop_length (int): Number of samples between successive frames. Default is 512.
    """
    unique_labels = np.unique(labels)
    num_labels = len(unique_labels)

    grid_size = int(np.ceil(np.sqrt(num_labels)))

    fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
    axes = axes.flatten()

    for i, label in enumerate(unique_labels):
        index = np.where(labels == label)[0][0]
        
        spectrogram = spectrograms[index]
        
        time_axis = np.arange(spectrogram.shape[1]) * hop_length / sr
        freq_axis = np.linspace(0, sr / 2, spectrogram.shape[0])

        axes[i].imshow(spectrogram, aspect='auto', origin='lower',
                       extent=[time_axis.min(), time_axis.max(), freq_axis.min(), freq_axis.max()])
        axes[i].set_title(f"Label: {label}", pad=5)
        axes[i].set_xlabel("Time (s)")
        axes[i].set_ylabel("Frequency (Hz)")
    
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout(pad=4.0)
    plt.show()
