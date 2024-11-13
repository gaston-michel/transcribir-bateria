from datetime import datetime
import os

from dataset import create_dataset, get_input_shape_from_dataset
from model import create_resnet, train_model, evaluate_model, plt_matrix, save_model, save_model_info
from preprocess import preprocess_wav_files

import config


def main():
    spectrograms, labels = preprocess_wav_files(
        "C:/Users/Gaston/Documents/Python/Datasets/Dataset",
        sr=config.SAMPLE_RATE,
        clip_size=config.CLIP_SIZE_MS,
        n_fft=config.FRAME_SIZE,
        hop_length=config.HOP_LENGTH,
        classes=config.CLASSES,
    )

    train_dataset, val_dataset, mlb = create_dataset(spectrograms, labels)

    input_shape = get_input_shape_from_dataset(train_dataset)
    num_classes = len(mlb.classes_)
    print(f"Classes: {mlb.classes_}")
    model = create_resnet(input_shape, num_classes)

    history = train_model(model, train_dataset, val_dataset, num_epochs=100)

    evaluate_model(model, val_dataset)
    plt_matrix(model, val_dataset, mlb)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_filename = f"model_{config.SAMPLE_RATE}_{config.CLIP_SIZE_MS}_{timestamp}.keras"
    save_model(model, model_filename)

    info_filename = os.path.splitext(model_filename)[0] + ".txt"
    save_model_info(model, mlb, history, info_filename)

    print(f"Model saved to: {model_filename}")
    print(f"Model info saved to: {info_filename}")

if __name__ == "__main__":
    main()