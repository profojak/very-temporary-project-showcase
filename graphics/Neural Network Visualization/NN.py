"""Visualization of Training of Neural Network - Jakub Pícha, Jakub Profota """

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppress TensorFlow warnings

import numpy as np
import keras as ks
import msgpack
import argparse
import json
import re


def blue_text(text):
    """Return the text in blue bold."""
    return f"\033[94m\033[1m{text}\033[0m"


def red_text(text):
    """Return the text in red bold."""
    return f"\033[91m\033[1m{text}\033[0m"


def white_text(text):
    """Return the text in white bold."""
    return f"\033[97m\033[1m{text}\033[0m"


class AfterEpochCallback(ks.callbacks.Callback):
    """Custom callback to save the state of the model after each epoch."""

    def __init__(self, x_test, y_test, export_path):
        super().__init__()
        self.x_test = x_test
        self.y_test = y_test
        self.export_path = export_path
        if export_path:
            if os.path.exists(export_path):
                os.remove(export_path)
            print(blue_text("Export file: ") + export_path)
            self.data = {"model":{}}

    def __del__(self):
        if self.export_path:
            byte = msgpack.packb(self.data)
            with open(self.export_path, "wb") as f:
                f.write(byte)

    def on_epoch_end(self, epoch, logs=None):
        """Save the model metadata after each epoch."""

        # Test the model on the test set
        score = self.model.evaluate(self.x_test, self.y_test, batch_size=1000,
                                    verbose=0)

        # Save the model metadata
        vals = self.model.predict(self.x_test, batch_size=1000, verbose=0)
        y_pred = vals.argmax(axis=-1)
        y_real = self.y_test.argmax(axis=-1)
        self.data["model"][epoch] = {
            "accuracy": score[1],
            "loss": score[0],
            "pred": y_pred.tolist(),
            "real": y_real.tolist(),
            "weights": {},
        }
        for i, v in enumerate(vals.T):
            self.data["model"][epoch]["weights"][str(i)] = v.tolist()


def load_mnist_data():
    """Load the MNIST dataset and preprocess it."""

    # Load the MNIST dataset
    (x_train, y_train), (x_test, y_test) = ks.datasets.mnist.load_data()
    # Normalize the data to the range [0, 1]
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255
    # Reshape the data to have shape (num_samples, 28, 28, 1)
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    # Convert class vectors to binary class matrices
    y_train = ks.utils.to_categorical(y_train, 10)
    y_test = ks.utils.to_categorical(y_test, 10)

    msg = f"MNIST: {x_train.shape[0]} training and {x_test.shape[0]} test "
    msg += f"{x_train.shape[1]}x{x_train.shape[2]} images"
    print(blue_text("Data: ") + msg)

    return (x_train, y_train), (x_test, y_test)


def test_model(model, mnist):
    """Test the model on the MNIST dataset."""

    (x_train, y_train), (x_test, y_test) = mnist

    # Compile again to suppress warnings when loading the model
    model.compile(loss="categorical_crossentropy", optimizer="adam",
                  metrics=["accuracy"])

    # Evaluate the model on the test set
    print(blue_text("Testing:"))
    score = model.evaluate(x_test, y_test, batch_size=1000, verbose=2)

    msg = str(round(score[1] * 100, 5)) + "% (" + str(score[1]) + ")"
    print(white_text(" Accuracy: ") + msg)
    print(white_text(" Loss: ") + str(score[0]))


def train_model(batch, epochs, mnist, save_path, export_path):
    """Train a simple neural network on the MNIST dataset."""

    (x_train, y_train), (x_test, y_test) = mnist

    # Build the model
    model = ks.Sequential([
        ks.Input(shape=(x_train.shape[1:])),
        ks.layers.Conv2D(32, kernel_size=(3, 3), activation="relu", name="c1"),
        ks.layers.MaxPooling2D(pool_size=(2, 2), name="p1"),
        ks.layers.Conv2D(64, kernel_size=(3, 3), activation="relu", name="c2"),
        ks.layers.MaxPooling2D(pool_size=(2, 2), name="p2"),
        ks.layers.Flatten(name="f"),
        ks.layers.Dropout(0.5, name="d"),
        ks.layers.Dense(10, activation="softmax", name="o"),
    ])

    model.summary()

    # Compile the model
    model.compile(loss="categorical_crossentropy", optimizer="adam",
                  metrics=["accuracy"])

    # Train the model
    print(blue_text("Training: ") + f"{batch} batch size, {epochs} epochs")
    callback = AfterEpochCallback(x_test, y_test, export_path)
    model.fit(x_train, y_train, batch_size=batch, epochs=epochs,
              validation_split=0.1, verbose=2, callbacks=[callback])

    # Save the model
    if save_path:
        model.save(save_path)
        print(blue_text("Save file: ") + save_path)

    return model


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="MNIST Neural Network",
        epilog="by Jakub Pícha, Jakub Profota",
        allow_abbrev=True,
    )
    parser.add_argument("mode", choices=["train", "test", "predict"],
                        default="train", help="Mode to run the script in")
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--save", type=str, default="model.keras", 
                       metavar="FILE", help="Path to save the trained model")
    group.add_argument("--load", type=str, default="model.keras",
                       metavar="FILE", help="Path to load the trained model")
    group.add_argument("--no-save", action="store_true",
                       help="Do not save the trained model")
    parser.add_argument("--epochs", type=int, default=15,
                        help="Number of epochs to train the model, " +
                        "higher value results in more accurate model")
    parser.add_argument("--batch", type=int, default=15000,
                        help="Batch size for training, " +
                        "higher value results in slower training")
    parser.add_argument("--export", type=str, default=None, metavar="FILE",
                        help="Metadata for visualization in MessagePack format")
    args = parser.parse_args()

    # Load the MNIST dataset
    mnist = load_mnist_data()

    # Train the model
    if args.mode == "train":
        if args.no_save:
            args.save = None
        model = train_model(args.batch, args.epochs, mnist,
                            args.save, args.export)
    elif args.mode == "test":
        if not os.path.exists(args.load):
            print(red_text("Error: ") + "Model file does not exist")
            exit(1)
        model = ks.models.load_model(args.load, compile=False)
        print(blue_text("Load file: ") + args.load)

    # Test the model
    test_model(model, mnist)
