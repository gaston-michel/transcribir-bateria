import librosa
import numpy as np
import config

def detect_onsets(audio: np.ndarray, threshold: float = config.ONSET_THRESHOLD) -> np.ndarray:
    """
    Detect onset times in the audio signal.

    Args:
        audio (np.ndarray): Audio time series.
        threshold (float): Detection threshold for onset detection.

    Returns:
        np.ndarray: Array of detected onset times in seconds.
    """
    onset_env = librosa.onset.onset_strength(y=audio, sr=config.SAMPLE_RATE, hop_length=config.HOP_LENGTH)
    
    onset_frames = librosa.onset.onset_detect(
        onset_envelope=onset_env,
        sr=config.SAMPLE_RATE,
        hop_length=config.HOP_LENGTH,
        delta=threshold,
        backtrack=config.ONSET_BACKTRACK,
        wait=int(config.ONSET_WAIT * config.SAMPLE_RATE / config.HOP_LENGTH)
    )

    onset_times = librosa.frames_to_time(
        onset_frames,
        sr=config.SAMPLE_RATE,
        hop_length=config.HOP_LENGTH
    )

    return onset_times