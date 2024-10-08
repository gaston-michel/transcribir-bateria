import librosa
import numpy as np
import config

def detect_beats(audio: np.ndarray) -> np.ndarray:
    """
    Detect beat times in the audio signal.

    Args:
        audio (np.ndarray): Audio time series.

    Returns:
        np.ndarray: Array of detected beat times in seconds.
    """
    onset_env = librosa.onset.onset_strength(y=audio, sr=config.SAMPLE_RATE)

    config.TEMPO, beat_frames = librosa.beat.beat_track(onset_envelope=onset_env, 
                                                 sr=config.SAMPLE_RATE,
                                                 start_bpm=100,
                                                 trim=False,
                                                 bpm=config.TEMPO,
                                                 tightness=100)

    beat_times = librosa.frames_to_time(beat_frames, sr=config.SAMPLE_RATE)

    return beat_times