# Audio Processing Settings
SAMPLE_RATE = 44100  # Hz
FRAME_SIZE = 1024
HOP_LENGTH = 256
CLIP_SIZE_MS = 200  # in milliseconds
CLIP_SIZE = int(CLIP_SIZE_MS * SAMPLE_RATE / 1000)
DATA_AUGMENTATION = False
APPLY_LOW_FILTER = False

# Model configuration
CLASSES = [
            'CR',
            # 'CR+HH',
            # 'CR+HH+KD',
            # 'CR+HH+KD+RD',
            # 'CR+HH+KD+SD',
            # 'CR+HH+SD',
            'CR+KD',
            # 'CR+KD+RD',
            'CR+KD+SD', 
            # 'CR+KD+T1', 
            'CR+SD',
            # 'CR+T3',
            'HH',
            'HH+KD',
            # 'HH+KD+RD',
            # 'HH+KD+RD+SD',
            'HH+KD+SD',
            # 'HH+KD+T1',
            # 'HH+KD+T2',
            # 'HH+KD+T3',
            # 'HH+RD',
            # 'HH+RD+SD',
            # 'HH+RD+T1',
            # 'HH+RD+T2',
            # 'HH+RD+T3',
            'HH+SD',
            # 'HH+SD+T1',
            # 'HH+SD+T2',
            # 'HH+SD+T3',
            # 'HH+T1',
            # 'HH+T2',
            # 'HH+T3',
            'KD',
            # 'KD+RD',
            # 'KD+RD+SD',
            # 'KD+RD+T1',
            # 'KD+RD+T2',
            # 'KD+RD+T3',
            'KD+SD',
            # 'KD+SD+T1',
            # 'KD+SD+T2',
            # 'KD+SD+T3',
            # 'KD+T1',
            # 'KD+T2',
            # 'KD+T3',
            # 'RD',
            # 'RD+SD',
            # 'RD+T1',
            # 'RD+T2',
            # 'RD+T3',
            'SD',
            # 'SD+T1',
            # 'SD+T2',
            # 'SD+T3',
            # 'T1',
            # 'T1+T2',
            # 'T1+T3',
            # 'T2',
            # 'T2+T3',
            # 'T3',
            # 'None',
          ]

# Export options
MODELS_DIR = 'models/'
MODELS_INFO_DIR = 'models_info/'