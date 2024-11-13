import os
import numpy as np
import scipy.signal as signal
import soundfile as sf

import config

def apply_lowpass_filter(audio: np.ndarray, cutoff_freq: float = 1000) -> np.ndarray:
    """
    Filter the audio signal using a lowpass filter with the specified cutoff frequency.

    Args:
        audio (np.ndarray): Audio time series.
        cutoff_freq (float): Cutoff frequency in Hz.

    Returns:
        np.ndarray: Array of filtered audio time series.
    """
    nyquist = 0.5 * config.SAMPLE_RATE
    normal_cutoff = cutoff_freq / nyquist
    b, a = signal.butter(5, normal_cutoff, btype='low', analog=False)
            
    return signal.filtfilt(b, a, audio, axis=0)

def apply_highpass_filter(audio: np.ndarray, cutoff_freq: float = 1000) -> np.ndarray:
    """
    Filter the audio signal using a highpass filter with the specified cutoff frequency.

    Args:
        audio (np.ndarray): Audio time series.
        cutoff_freq (float): Cutoff frequency in Hz.

    Returns:
        np.ndarray: Array of filtered audio time series.
    """
    nyquist = 0.5 * config.SAMPLE_RATE
    normal_cutoff = cutoff_freq / nyquist
    b, a = signal.butter(5, normal_cutoff, btype='high', analog=False)
            
    return signal.filtfilt(b, a, audio, axis=0)