import io
import librosa
import numpy as np
import config

def load_audio(audio_data: bytes) -> np.ndarray:
    """
    Load an audio file from bytes, slice it from a specified start time, and return it as a numpy array.

    Args:
        audio_data (bytes): Audio file data in bytes.

    Returns:
        np.ndarray: Sliced audio time series as a numpy array.
    """
    audio_file = io.BytesIO(audio_data)
    audio, _ = librosa.load(audio_file, sr=config.SAMPLE_RATE)
    audio = slice_audio_data(audio, config.START_TIME)
    return audio

def slice_audio_data(audio_data: np.ndarray, 
                     start_time: float, 
                     sample_rate: int = config.SAMPLE_RATE
                     ) -> np.ndarray:
    """
    Slices the audio data from a given start time.

    Parameters:
    audio_data (numpy.ndarray): The array containing audio samples.
    start_time (float): The start time in seconds from which to slice the audio.
    sample_rate (int): The sample rate of the audio data (samples per second).

    Returns:
    numpy.ndarray: The sliced audio data starting from the specified start time.
    """
    start_sample = int(start_time * sample_rate)
    return audio_data[start_sample:]