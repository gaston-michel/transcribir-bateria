# Audio Processing Settings
SAMPLE_RATE = 44100  # Hz
FRAME_SIZE = 1024
HOP_LENGTH = 256
TEMPO = None
START_TIME = 0.0 # seconds

# Onset Detection
ONSET_THRESHOLD = 0.1
ONSET_WAIT = 0.1  # seconds
ONSET_BACKTRACK = True

# Beat Detection
MIN_TEMPO = 60  # BPM
MAX_TEMPO = 200  # BPM

# Beat Alignment
ALIGNMENT_TOLERANCE = 0.2 # seconds

# Drum Part Prediction
CLASSES_LABELS = [
                ['CR'], 
                ['HH'], 
                ['KD'], 
                ['SD'],
                ]
PREDICTION_THRESHOLD = 0.7
NONE_PREDICTION_THRESHOLD = 0.7
CLIP_SIZE_MS = 200  # miliseconds
CLIP_SIZE = int(CLIP_SIZE_MS * SAMPLE_RATE / 1000)

# Sheet Music Generation
FILE_NAME = None
TIME_SIGNATURE = "4/4"  # 4/4 time

# File Paths
MODELS_DIR = 'models/'
DRUM_MODEL_PATH = MODELS_DIR + 'model_44100_200_20240928_215455.keras'
