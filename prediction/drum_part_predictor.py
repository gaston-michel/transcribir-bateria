import numpy as np
import librosa
from sklearn.preprocessing import MultiLabelBinarizer
import tensorflow as tf
from typing import List, Tuple
import config

class DrumPartPredictor:
    def __init__(self, model_path: str = config.DRUM_MODEL_PATH):
        self.model = tf.keras.models.load_model(model_path)
        self.mlb = MultiLabelBinarizer().fit(config.CLASSES_LABELS)

    def calculate_spectrograms(self, audio: np.ndarray, onset_times: np.ndarray) -> List[np.ndarray]:
        """Calculate spectrograms from the audio at each onset."""
        return [self._calculate_single_spectrogram(audio, onset) for onset in onset_times]

    def _calculate_single_spectrogram(self, audio: np.ndarray, onset: float) -> np.ndarray:
        """Calculate spectrogram for a single onset."""
        clip = self._extract_audio_clip(audio, onset)
        return self._clip_to_spectrogram(clip)

    def _extract_audio_clip(self, audio: np.ndarray, onset: float) -> np.ndarray:
        """Extract an audio clip of the specified onset."""
        start = librosa.time_to_samples(onset, sr=config.SAMPLE_RATE)
        end = min(start + config.CLIP_SIZE, len(audio))
        start = max(0, end - config.CLIP_SIZE)
        return audio[start:end]

    def _clip_to_spectrogram(self, clip: np.ndarray) -> np.ndarray:
        """Convert a WAV audio clip to a normalized Mel spectrogram."""
        spectrogram = librosa.feature.melspectrogram(
            y=clip, 
            sr=config.SAMPLE_RATE, 
            n_fft=config.FRAME_SIZE, 
            hop_length=config.HOP_LENGTH
        )
        spectrogram_db = librosa.amplitude_to_db(abs(spectrogram))
        return np.array([spectrogram_db / np.max(spectrogram_db)])

    def predict(self, spectrograms: List[np.ndarray]) -> List[Tuple[str, ...]]:
        """Predict drum parts based on spectrogram features."""
        predictions = self.model.predict(np.vstack(spectrograms))
        thresholded_predictions = (predictions > config.PREDICTION_THRESHOLD).astype(int)
        return [self._decode_prediction(pred) for pred in thresholded_predictions]

    def _decode_prediction(self, prediction: np.ndarray) -> Tuple[str, ...]:
        """Decode a single prediction."""
        decoded = self.mlb.inverse_transform(prediction.reshape(1, -1))[0]
        return ('None',) if len(decoded) == 0 else decoded

def predict_drum_parts(audio: np.ndarray, onset_times: np.ndarray) -> List[Tuple[str, ...]]:
    """Predict drum parts for the given audio at the specified onset times."""
    predictor = DrumPartPredictor()
    spectrograms = predictor.calculate_spectrograms(audio, onset_times)
    return predictor.predict(spectrograms)