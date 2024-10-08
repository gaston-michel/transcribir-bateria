import numpy as np
import config


def align_beats_with_onsets(beat_times: np.ndarray, 
                            onset_times: np.ndarray, 
                            tolerance: float = config.ALIGNMENT_TOLERANCE) -> np.ndarray:
    """
    Align detected beats with onset times, removing beats before the first onset.

    Args:
        beat_times (np.ndarray): Array of beat times in seconds.
        onset_times (np.ndarray): Array of onset times in seconds.
        tolerance (float): Maximum time difference to consider a beat aligned with an onset.

    Returns:
        np.ndarray: Array of aligned beat times in seconds.
    """
    def filter_beats_before_first_onset(beat_times: np.ndarray, first_onset: float) -> np.ndarray:
        return beat_times[beat_times >= first_onset]
    
    def get_closest_onset(beat: float, onset_times: np.ndarray) -> float:
        return onset_times[np.argmin(np.abs(onset_times - beat))]
    
    def is_within_tolerance(beat: float, onset: float, tolerance: float) -> bool:
        return np.abs(onset - beat) <= tolerance

    first_onset = onset_times[0]
    valid_beats = filter_beats_before_first_onset(beat_times, first_onset)
    
    aligned_beats = [
        get_closest_onset(beat, onset_times) if is_within_tolerance(beat, get_closest_onset(beat, onset_times), tolerance) else beat
        for beat in valid_beats
    ]
    
    return np.array(aligned_beats)
