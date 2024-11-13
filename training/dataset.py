import tensorflow as tf
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer


def create_dataset(spectrograms, labels, validation_split=0.2, batch_size=32):
    """
    Prepares the data and creates TensorFlow datasets from spectrograms and labels.

    Args:
        spectrograms (numpy.ndarray): Array of spectrograms.
        labels (numpy.ndarray): Array of corresponding labels.
        validation_split (float): Fraction of data to use for validation.
        batch_size (int): Batch size for the datasets.

    Returns:
        tuple: A tuple containing the training and validation datasets, and the label encoder.
    """
    x_train, x_val, y_train, y_val, mlb = prepare_data(spectrograms, labels, validation_split)
    train_dataset = create_tf_dataset(x_train, y_train, shuffle=True, batch_size=batch_size)
    val_dataset = create_tf_dataset(x_val, y_val, shuffle=False, batch_size=batch_size)
    return train_dataset, val_dataset, mlb


def get_input_shape_from_dataset(train_dataset):
    """
    Determines the input shape for the model from the training dataset.

    Args:
        train_dataset (tf.data.Dataset): The training dataset.

    Returns:
        tuple: The shape of the input data, excluding the batch size.
    """
    for batch in train_dataset.take(1):
        input_shape = batch[0].shape[1:]
        return input_shape


def prepare_data(spectrograms, labels, validation_split=0.2):
    """Encodes labels and splits data into training and validation sets."""
    spectrograms = add_channel_dimension(spectrograms)
    mlb = encode_labels(labels)
    labels_split = [label.split('+') for label in labels]
    labels_encoded = mlb.transform(labels_split)
    x_train, x_val, y_train, y_val = split_data(spectrograms, labels_encoded, validation_split)
    return x_train, x_val, y_train, y_val, mlb


def create_tf_dataset(features, labels, shuffle, batch_size):
    """Creates a TensorFlow dataset from features and labels, with optional shuffling."""
    dataset = tf.data.Dataset.from_tensor_slices((features, labels))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(features))
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def add_channel_dimension(spectrograms):
    """Adds a channel dimension to the spectrograms."""
    return np.expand_dims(spectrograms, -1)


def encode_labels(labels):
    """Encodes labels using MultiLabelBinarizer."""
    mlb = MultiLabelBinarizer()
    labels_split = [label.split('+') for label in labels]
    mlb.fit(labels_split)
    return mlb


def split_data(features, labels, validation_split):
    """Splits data into training and validation sets."""
    indices = np.arange(len(features))
    np.random.shuffle(indices)
    
    x_data = features[indices]
    y_data = labels[indices]
    
    train_size = int((1 - validation_split) * len(x_data))
    x_train, x_val = x_data[:train_size], x_data[train_size:]
    y_train, y_val = y_data[:train_size], y_data[train_size:]
    
    return x_train, x_val, y_train, y_val