from contextlib import redirect_stdout
import io
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import multilabel_confusion_matrix
import seaborn as sns

import config


def create_model(input_shape, num_classes):
    """Creates and compiles a CNN model for audio classification."""
    model = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(shape=input_shape),
            
            # First Convolutional Block
            tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D((2, 2)),

            # Second Convolutional Block
            tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D((2, 2)),

            # Third Convolutional Block
            tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D((2, 2)),

            # # Fourth Convolutional Block
            # tf.keras.layers.Conv2D(256, (3, 3), activation="relu"),
            # tf.keras.layers.BatchNormalization(),
            # tf.keras.layers.MaxPooling2D((2, 2)),

            # Global Average Pooling instead of Flatten
            tf.keras.layers.GlobalAveragePooling2D(),

            # Fully Connected Layer
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.5),

            # Output Layer
            tf.keras.layers.Dense(num_classes, activation="sigmoid"),
        ]
    )

    model.compile(
        optimizer="adam", loss="binary_crossentropy", 
        metrics=["accuracy", "precision", "recall", "f1_score"]
    )
    return model


def residual_block(x, filters, kernel_size=3, stride=1):
    """A residual block with two convolutional layers and a matching shortcut."""
    # Shortcut (Identity Connection)
    shortcut = x

    # First Conv layer
    x = tf.keras.layers.Conv2D(filters, kernel_size, strides=stride, padding='same')(x)
    # x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    # Second Conv layer
    x = tf.keras.layers.Conv2D(filters, kernel_size, strides=1, padding='same')(x)
    # x = tf.keras.layers.BatchNormalization()(x)

    # Adjust the shortcut dimensions if the input and output have different shapes
    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = tf.keras.layers.Conv2D(filters, (1, 1), strides=stride, padding='same')(shortcut)
        # shortcut = tf.keras.layers.BatchNormalization()(shortcut)

    # Adding the shortcut connection
    x = tf.keras.layers.Add()([x, shortcut])
    x = tf.keras.layers.ReLU()(x)
    
    return x


def create_resnet(input_shape, num_classes):
    """Creates a ResNet-like model for audio classification."""
    inputs = tf.keras.Input(shape=input_shape)

    # Initial Conv Layer
    x = tf.keras.layers.Conv2D(64, (7, 7), strides=2, padding='same')(inputs)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPooling2D((3, 3), strides=2, padding='same')(x)

    # Residual Blocks
    x = residual_block(x, 64)  # Residual Block 1
    x = residual_block(x, 64)  # Residual Block 2

    x = residual_block(x, 128, stride=2)  # Residual Block 3
    x = residual_block(x, 128)            # Residual Block 4

    x = residual_block(x, 256, stride=2)  # Residual Block 5
    x = residual_block(x, 256)            # Residual Block 6

    # Global Average Pooling
    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    # Fully Connected Layer
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)

    # Output Layer
    outputs = tf.keras.layers.Dense(num_classes, activation='sigmoid')(x)

    # Model Creation
    model = tf.keras.Model(inputs, outputs)
    
    model.compile(
        optimizer="adam", loss="binary_crossentropy", 
        metrics=["accuracy", "precision", "recall", "f1_score"]
    )
    return model


def train_model(model, train_dataset, val_dataset, num_epochs=50):
    """Trains the model and returns the training history."""
    lr_schedule = tf.keras.callbacks.LearningRateScheduler(
        lambda epoch: 1e-3 * 10 ** (-epoch / 20)
    )
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=num_epochs,
        callbacks=[early_stopping, lr_schedule],
    )
    return history


def evaluate_model(model, val_dataset):
    """Evaluates the model and prints the evaluation metrics."""
    loss, accuracy, precision, recall, f1_score = model.evaluate(val_dataset)
    print(f"Evaluation Metrics:")
    print(f"Loss: {loss:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    for i, f1 in enumerate(f1_score):
        print(f"Class {i} - F1-Score: {f1:.4f}")


def plt_matrix(model, val_dataset, mlb, threshold=0.5):
    """Evaluates the model and plots a confusion matrix for multi-label classification."""
    predicted_labels = []
    expected_labels = []

    for batch in val_dataset:
        x_val, y_val = batch
        predictions = model.predict(x_val)
        
        predicted_labels.extend((predictions > threshold).astype(int))
        expected_labels.extend(y_val.numpy())

    predicted_labels = np.array(predicted_labels)
    expected_labels = np.array(expected_labels)

    conf_matrices = multilabel_confusion_matrix(expected_labels, predicted_labels)

    for i, conf_matrix in enumerate(conf_matrices):
        plt.figure(figsize=(6, 5))
        sns.heatmap(
            conf_matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Not " + mlb.classes_[i], mlb.classes_[i]],
            yticklabels=["Not " + mlb.classes_[i], mlb.classes_[i]],
        )
        plt.xlabel("Predicted Labels")
        plt.ylabel("True Labels")
        plt.title(f"Confusion Matrix for {mlb.classes_[i]}")
        plt.show()


def get_model_summary(model):
    """Returns the model summary as a string."""
    stream = io.StringIO()
    with redirect_stdout(stream):
        model.summary()
    summary_str = stream.getvalue()
    return summary_str


def save_model(model, filename):
    """Saves the trained model to a file."""
    if not os.path.exists(config.MODELS_DIR):
        os.makedirs(config.MODELS_DIR)
    model.save(os.path.join(config.MODELS_DIR, filename))


def save_model_info(model, mlb, history, filename):
    """Saves the model configuration, summary, and training results to a file."""
    model_config = [
        f"Sample Rate: {config.SAMPLE_RATE} Hz",
        f"Clip Size: {config.CLIP_SIZE_MS} ms",
        f"n_fft: {config.FRAME_SIZE}",
        f"hop_length: {config.HOP_LENGTH}",
        f"Data Augmentation: {config.DATA_AUGMENTATION}",
        f"Apply Low Filter: {config.APPLY_LOW_FILTER}",
        f"Classes: {', '.join(mlb.classes_)}"
    ]
    model_summary = get_model_summary(model)

    training_results = [
        f"Number of Epochs: {len(history.history['loss'])}",
        f"Final Accuracy: {history.history['accuracy'][-1]:.4f}",
        f"Final Precision: {history.history['precision'][-1]:.4f}",
        f"Final Recall: {history.history['recall'][-1]:.4f}",
    ]
    for i, f1 in enumerate(history.history['f1_score'][-1]):
        training_results.append(f"Final F1-Score for Class {i}: {f1:.4f}")

    if not os.path.exists(config.MODELS_INFO_DIR):
        os.makedirs(config.MODELS_INFO_DIR)
    with open(os.path.join(config.MODELS_INFO_DIR, filename), "w", encoding="utf-8") as f:
        f.write("Model Configuration:\n")
        f.write("\n".join(model_config))
        f.write("\n\nModel Summary:\n")
        f.write(model_summary)
        f.write("\n\nTraining Results:\n")
        f.write("\n".join(training_results))