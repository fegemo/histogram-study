import argparse
import os
from math import ceil

import io_utils

SEED = 42

DATASET_NAMES = [f"pixel-landscapes", f"photo-landscapes", "pixelart-misc"]
DATA_FOLDERS = [
    os.sep.join(["datasets", folder])
    for folder
    in DATASET_NAMES
]

DATASET_MASK = [1, 0, 0]
DATASET_SIZES = [798, 3821, 19481]
DATASET_SIZES = [n*m for n, m in zip(DATASET_SIZES, DATASET_MASK)]

DATASET_SIZE = sum(DATASET_SIZES)
TRAIN_PERCENTAGE = 0.85
TRAIN_SIZES = [ceil(n * TRAIN_PERCENTAGE) for n in DATASET_SIZES]
TRAIN_SIZE = sum(TRAIN_SIZES)
TEST_SIZES = [DATASET_SIZES[i] - TRAIN_SIZES[i]
              for i, n in enumerate(DATASET_SIZES)]
# TEST_SIZES = [0, 0, 44, 0, 0]
TEST_SIZE = sum(TEST_SIZES)

BUFFER_SIZE = DATASET_SIZE
BATCH_SIZE = 4

IMG_SIZE = 64
INPUT_CHANNELS = 3
OUTPUT_CHANNELS = 3

TEMP_FOLDER = "temp-study"


class SingletonMeta(type):
    """
    The Singleton class can be implemented in different ways in Python. Some
    possible methods include: base class, decorator, metaclass. We will use the
    metaclass because it is best suited for this purpose.
    """

    _instances = {}

    def __call__(cls, *args, **kwargs):
        """
        Possible changes to the value of the `__init__` argument do not affect
        the returned instance.
        """
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]


class OptionParser(metaclass=SingletonMeta):
    def __init__(self) -> None:
        self.parser = argparse.ArgumentParser()
        self.initialized = False
        self.values = {}

    def initialize(self):
        self.parser.add_argument(
            "dataset", help="one from { pixel-landscapes, photo-landscapes-landscapes, pixel-landscapes } - the dataset to train with")
        self.parser.add_argument(
            "--model", help="one from { vanilla, akash } the network architecture to use", default="akash")
        self.parser.add_argument("--verbose", help="outputs verbosity information", default=False, action="store_true")
        self.parser.add_argument("--batch", type=int, help="the batch size", default=32)
        self.parser.add_argument("--latent-dim", type=int, help="the latent code dimension size", default=128)
        self.parser.add_argument("--crop-size", type=int, help="the size with which to work with images", default=32)
        self.parser.add_argument("--channels", type=int, help="the number of channels with which to represent images. "
                                                              "Use either 1 or 3 (default)", default=3)
        self.parser.add_argument("--downsamples", type=int, help="the number of down sampling steps in the encoder "
                                                                 "(2 to more)", default=2)
        self.parser.add_argument("--lambda-kl", type=float, help="weight for the KL loss", default=.1)
        self.parser.add_argument("--lambda-bce", type=float, help="weight for the BCE loss", default=0.)
        self.parser.add_argument("--lambda-mse", type=float, help="weight for the MSE loss", default=1.)
        self.parser.add_argument("--lambda-ssim", type=float, help="weight for the SSIM loss", default=0.)
        self.parser.add_argument("--lambda-histogram", type=float, help="weight for the histogram loss", default=1.)
        self.parser.add_argument("--d-steps", type=int, help="critic steps for every generator in wgan", default=5)
        self.parser.add_argument("--lambda-gp", type=float, help="weight for the gradient penalty (wgan)", default=10.)
        self.parser.add_argument("--lr1", type=float, help="main learning rate", default=0.0002)
        self.parser.add_argument("--lr2", type=float, help="secondary learning rate", default=0.001)
        self.parser.add_argument("--epochs", type=int, help="number of epochs to train", default=100)

        self.parser.add_argument("--description", help="string description for tensorboard", default="")
        self.parser.add_argument(
            "--log-folder", help="the folder in which the training procedure saves the logs", default=TEMP_FOLDER)
        self.initialized = True

    def parse(self, return_parser=False):
        if not self.initialized:
            self.initialize()
        self.values = self.parser.parse_args()
        setattr(self.values, "seed", SEED)
        dataset_path = DATA_FOLDERS[["pixel-landscapes", "photo-landscapes-landscapes", "pixel-landscapes"].index(self.values.dataset)]
        setattr(self.values, "dataset", dataset_path)

        if return_parser:
            return self.values, self
        else:
            return self.values

    def get_description(self, param_separator=",", key_value_separator="-"):
        sorted_args = sorted(vars(self.values).items())
        description = param_separator.join(map(lambda p: f"{p[0]}{key_value_separator}{p[1]}", sorted_args))
        return description

    def save_configuration(self, folder_path):
        io_utils.ensure_folder_structure(folder_path)
        with open(os.sep.join([folder_path, "configuration.txt"]), "w") as file:
            file.write(self.get_description("\n", ": ") + "\n")


def in_notebook():
    try:
        from IPython import get_ipython
        if "IPKernelApp" not in get_ipython().config:  # pragma: no cover
            return False
    except ImportError:
        return False
    except AttributeError:
        return False
    return True


if not in_notebook():
    options = OptionParser().parse()

    BATCH_SIZE = options.batch
    TEMP_FOLDER = options.log_folder
