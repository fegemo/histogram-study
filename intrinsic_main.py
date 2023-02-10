import intdim_mle
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import tensorflow as tf


def calculate_intdim(images):
    k1 = 10  # start of interval(included)
    k2 = 20  # end of interval(included)
    intdim_k_repeated = intdim_mle.repeated(intdim_mle.intrinsic_dim_scale_interval,
                                            images,
                                            mode="bootstrap",
                                            nb_iter=100,  # nb_iter for bootstrapping
                                            verbose=1,
                                            k1=k1, k2=k2)
    intdim_k_repeated = np.array(intdim_k_repeated)
    # the shape of intdim_k_repeated is (nb_iter, size_of_interval) where
    # nb_iter is number of bootstrap iterations (here 500) and size_of_interval
    # is (k2 - k1 + 1).
    # Plotting the histogram of intrinsic dimensionality estimations repeated over
    # nb_iter experiments
    plt.hist(intdim_k_repeated.mean(axis=1))

    mean = intdim_k_repeated.mean()
    std = intdim_k_repeated.std()
    return mean, std


def load_cropped_dataset_128(dataset_path, interpolation):
    return tf.keras.utils.image_dataset_from_directory(
        dataset_path,
        labels=None,
        batch_size=None,
        image_size=(128, 128),
        interpolation=interpolation,
        crop_to_aspect_ratio=True
    )


dataset_names = ["pixel-landscapes", "photo-landscapes", "pixelart-misc"]
dataset_folders = [f"datasets/{name}/" for name in dataset_names]
print(dataset_folders)

for dataset_folder, dataset_name in zip(dataset_folders, dataset_names):
    print(f"Calculating intrinsic dimensionality of {dataset_name}")
    interpolation = "nearest" if "pixel" in dataset_name else "bilinear"
    ic = np.array(list(load_cropped_dataset_128(dataset_folder, interpolation).as_numpy_iterator()))
    ic = ic / 255.
    # ic = np.array(io.ImageCollection(dataset_folder))
    print("Images found:", ic.shape[0])
    ic = ic.reshape((ic.shape[0]), -1)
    intrinsic_dim, std = calculate_intdim(ic)
    print(f"Intrinsic dimensionality for {dataset_name} (mean/std):", f"{intrinsic_dim:.4f} ({std:.2f})")
