import tensorflow as tf

import io_utils
from configuration import *


def normalize(image):
    """
    Turns an image from the [0, 255] range into [-1, 1], keeping the same data type.
    Parameters
    ----------
    image a tensor representing an image
    Returns the image in the [-1, 1] range
    -------
    """
    return (image / 127.5) - 1


def denormalize(image):
    """
    Turns an image from the [-1, 1] range into [0, 255], keeping the same data type.
    Parameters
    ----------
    image a tensor representing an image
    Returns the image in the [0, 255] range
    -------
    """
    return (image + 1) * 127.5


def load_rgb_dataset(path, options, train_percentage=0.85):
    crop_size = options.crop_size

    # loads an image from the file system and transforms it for the network:
    # (a) casts to float, (b) ensures transparent pixels are black-transparent, and (c)
    # puts the values in the range of [-1, 1]
    def load_image(image_path):
        channels = options.channels
        try:
            image = tf.io.read_file(image_path)
            image = tf.image.decode_image(image, channels=3, expand_animations=False)
            image = tf.cast(image, "float32")
            if channels == 1:
                image = tf.image.rgb_to_grayscale(image)
            image = normalize(image)
            # image = image / 255.
        except UnicodeDecodeError:
            print("Error opening image in ", image_path)
        return image

    def make_prepare_image_func(should_augment=True):
        def prepare_image(image):
            height, width, channels = tf.shape(image)[0], tf.shape(image)[1], tf.shape(image)[2]

            height_pad = tf.maximum(0, (crop_size - height))
            width_pad = tf.maximum(0, (crop_size - width))
            needs_padding = height_pad > 0 or width_pad > 0
            needs_cropping = height > crop_size or width > crop_size

            # 1. padding
            if needs_padding:
                # cannot REFLECT-pad too much (should not exceed original size)
                # so we do it as many times as it is necessary to fill the crop_size
                while height_pad > 0 or width_pad > 0:
                    height_pad_top = tf.minimum(height - 1, height_pad // 2)
                    height_pad_bottom = tf.minimum(height - 1, tf.cast(height_pad, tf.int32) - (height_pad // 2))
                    width_pad_left = tf.minimum(width - 1, width_pad // 2)
                    width_pad_right = tf.minimum(width - 1, tf.cast(width_pad, tf.int32) - (width_pad // 2))
                    paddings = [
                        [height_pad_top, height_pad_bottom],
                        [width_pad_left, width_pad_right],
                        [0, 0]]
                    image = tf.pad(image, paddings, "REFLECT")
                    height_pad -= height_pad_top + height_pad_bottom
                    width_pad -= width_pad_left + width_pad_right

            # 2. random crop
            if needs_cropping:
                size = [crop_size, crop_size, channels]
                if should_augment:
                    image = tf.image.random_crop(image, size=size)
                else:
                    image = tf.image.stateless_random_crop(image, size=size, seed=(options.seed, options.seed))

            return image

        return prepare_image

    ds = tf.data.Dataset.list_files([os.path.join(path, f"*.{extension}") for extension in ["png", "jpg", "gif"]],
                                    shuffle=False) \
        .map(load_image, num_parallel_calls=tf.data.AUTOTUNE)

    no_train_examples = tf.cast(tf.math.ceil(train_percentage * tf.cast(ds.cardinality(), tf.float32)), tf.int64)
    no_test_examples = ds.cardinality() - no_train_examples

    train_ds = ds.map(make_prepare_image_func(True), num_parallel_calls=tf.data.AUTOTUNE)\
        .take(no_train_examples)
    test_ds = ds.map(make_prepare_image_func(False), num_parallel_calls=tf.data.AUTOTUNE)\
        .skip(no_train_examples)\
        .take(no_test_examples)

    return train_ds, test_ds
