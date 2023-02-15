import os.path

import tensorflow as tf
from tensorflow import keras
from keras import layers
import matplotlib.pyplot as plt

import histogram
from base_model import BaseModel
from io_utils import ensure_folder_structure


class CVAE(BaseModel):

    def __init__(self, options, **kwargs):
        super().__init__(**kwargs)
        self.options = options
        self.max_filters_in_a_layer = 2048
        self.latent_dim = options.latent_dim

        if options.model == "akash":
            self.encoder = self.create_encoder()
            self.decoder = self.create_decoder()
        else:
            self.encoder = self.create_encoder_tfkeras()
            self.decoder = self.create_decoder_tfkeras()
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.bce_tracker = keras.metrics.Mean(name="bce_loss")
        self.kl_tracker = keras.metrics.Mean(name="kl_loss")
        self.mse_tracker = keras.metrics.Mean(name="mse_loss")
        self.mae_tracker = keras.metrics.Mean(name="mae_loss")
        self.ssim_tracker = keras.metrics.Mean(name="ssim_loss")
        self.histogram_tracker = keras.metrics.Mean(name="histogram_loss")

    @property
    def models(self):
        return [self.encoder, self.decoder]

    def create_encoder(self):
        size = self.options.crop_size
        downsamples = self.options.downsamples

        encoder_inputs = keras.Input(shape=(size, size, self.options.channels))
        x = encoder_inputs

        filters = 64

        # PixelSight block
        x = layers.Conv2D(filters, kernel_size=1, strides=1, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

        for i in range(downsamples):
            x = layers.Conv2D(filters, kernel_size=4, strides=2, padding="same", use_bias=False)(x)
            x = layers.BatchNormalization()(x)
            x = layers.ReLU()(x)
            filters = tf.minimum(filters * 2, self.max_filters_in_a_layer)

        x = layers.Flatten()(x)

        z_mean = layers.Dense(self.latent_dim, name="z_mean")(x)
        z_log_var = layers.Dense(self.latent_dim, name="z_log_var")(x)
        z = Sampling()([z_mean, z_log_var])

        encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
        return encoder

    def create_decoder(self):
        size = self.options.crop_size
        upsamples = self.options.downsamples
        encoder_output_size = size // (2 ** upsamples)
        channels = self.options.channels

        latent_inputs = keras.Input(shape=(self.latent_dim,))
        x = latent_inputs
        x = layers.Dense(encoder_output_size * encoder_output_size * self.options.latent_dim, activation="relu")(x)
        x = layers.Reshape((encoder_output_size, encoder_output_size, self.options.latent_dim))(x)

        filters = 2 ** (6 + upsamples - 1)
        for i in range(upsamples):
            capped_filters = tf.cast(tf.minimum(filters, self.max_filters_in_a_layer), tf.int32)
            x = layers.Conv2DTranspose(capped_filters, kernel_size=4, strides=2, padding="same", use_bias=False)(x)
            x = layers.BatchNormalization()(x)
            x = layers.ReLU()(x)
            filters /= 2

        decoder_outputs = layers.Conv2DTranspose(channels, kernel_size=1, activation="tanh", padding="same")(x)
        decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
        return decoder

    def create_encoder_tfkeras(self):
        size = self.options.crop_size
        downsamples = self.options.downsamples

        encoder_inputs = keras.Input(shape=(size, size, self.options.channels))
        x = encoder_inputs

        filters = 512
        for i in range(downsamples):
            x = layers.Conv2D(filters, kernel_size=2, strides=2, padding="same")(x)
            x = layers.LeakyReLU()(x)
            filters = tf.minimum(filters * 2, self.max_filters_in_a_layer)

        x = layers.Flatten()(x)

        z_mean = layers.Dense(self.latent_dim, name="z_mean")(x)
        z_log_var = layers.Dense(self.latent_dim, name="z_log_var")(x)
        z = Sampling()([z_mean, z_log_var])

        encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
        return encoder

    def create_decoder_tfkeras(self):
        size = self.options.crop_size
        upsamples = self.options.downsamples
        encoder_output_size = size // (2 ** upsamples)
        channels = self.options.channels

        latent_inputs = keras.Input(shape=(self.latent_dim,))
        x = latent_inputs
        x = layers.Dense(encoder_output_size * encoder_output_size * self.options.latent_dim, activation="relu")(x)
        x = layers.Reshape((encoder_output_size, encoder_output_size, self.options.latent_dim))(x)

        filters = 2 ** (9 + upsamples - 1)
        for i in range(upsamples):
            capped_filters = tf.cast(tf.minimum(filters, self.max_filters_in_a_layer), tf.int32)
            x = layers.Conv2DTranspose(capped_filters, kernel_size=2, strides=2, padding="same")(x)
            x = layers.LeakyReLU()(x)
            filters /= 2

        decoder_outputs = layers.Conv2DTranspose(channels, kernel_size=2, activation="tanh", padding="same")(x)
        decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
        return decoder

    def call(self, batch):
        _, _, _, reconstruction = self._compute_prediction(batch)
        return reconstruction

    def _compute_prediction(self, batch):
        z_mean, z_log_var, encoded = self.encoder(batch)
        decoded = self.decoder(encoded)
        return z_mean, z_log_var, encoded, decoded

    def _compute_loss(self, batch, reconstruction, z_mean, z_log_var):
        bce_loss = tf.reduce_mean(
            tf.reduce_sum(
                keras.losses.binary_crossentropy(batch / 2. + 0.5, reconstruction / 2. + 0.5), axis=(1, 2)
            )
        )
        kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))

        mse_loss = tf.reduce_mean(tf.math.squared_difference(reconstruction, batch))
        mae_loss = tf.reduce_mean(tf.abs(reconstruction - batch))
        ssim_loss = 1. - tf.reduce_mean(tf.image.ssim(batch, reconstruction, max_val=2.))

        original_histogram = histogram.calculate_rgbuv_histogram(batch)
        reconstructed_histogram = histogram.calculate_rgbuv_histogram(reconstruction)
        histogram_loss = histogram.hellinger_loss(original_histogram, reconstructed_histogram)

        total_loss = \
            self.options.lambda_bce * bce_loss + \
            self.options.lambda_kl * kl_loss + \
            self.options.lambda_mse * mse_loss + \
            self.options.lambda_ssim * ssim_loss + \
            self.options.lambda_histogram * histogram_loss

        self.bce_tracker.update_state(bce_loss)
        self.kl_tracker.update_state(kl_loss)
        self.mse_tracker.update_state(mse_loss)
        self.mae_tracker.update_state(mae_loss)
        self.ssim_tracker.update_state(ssim_loss)
        self.histogram_tracker.update_state(histogram_loss)
        self.total_loss_tracker.update_state(total_loss)

        return total_loss

    @tf.function
    def train_step(self, batch):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z, reconstruction = self._compute_prediction(batch)
            total_loss = self._compute_loss(batch, reconstruction, z_mean, z_log_var)

        gradients = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        return {m.name: m.result() for m in self.metrics}

    @tf.function
    def test_step(self, batch):
        z_mean, z_log_var, z, reconstruction = self._compute_prediction(batch)
        self._compute_loss(batch, reconstruction, z_mean, z_log_var)

        return {m.name: m.result() for m in self.metrics}

    @property
    def metrics(self):
        return [
            self.bce_tracker,
            self.kl_tracker,
            self.mse_tracker,
            self.mae_tracker,
            self.ssim_tracker,
            self.histogram_tracker,
            self.total_loss_tracker
        ]


class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z"""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch_size, dim = tf.shape(z_mean)[0], tf.shape(z_mean)[1]
        epsilon = tf.random.uniform(shape=(batch_size, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class PreviewGeneratedImagesCallback(keras.callbacks.Callback):
    def __init__(self, log_folder, every_n_epochs, train_ds, test_ds, no_images=8):
        super().__init__()
        self.every_n_epochs = every_n_epochs
        self.log_folder = log_folder
        self.last_epoch_that_generated = -1
        self.last_epoch_watched = -1

        self.train_examples = next(iter(train_ds.take(no_images).batch(no_images)))
        self.test_examples = next(iter(test_ds.take(no_images).batch(no_images)))
        self.no_images = no_images

        self.text_size = 24

    def on_epoch_end(self, epoch, logs):
        if epoch == 0 or (epoch + 1) % self.every_n_epochs == 0:
            self.generate_images(epoch)
            self.last_epoch_that_generated = epoch
        self.last_epoch_watched = epoch

    def on_train_end(self, logs):
        if self.last_epoch_that_generated < self.last_epoch_watched:
            self.generate_images(self.last_epoch_watched)
            self.last_epoch_that_generated = self.last_epoch_watched

    def generate_images(self, epoch):
        model = self.model
        ensure_folder_structure(os.sep.join([self.log_folder, "images"]))

        for images, name in zip([self.train_examples, self.test_examples], ["train", "test"]):
            _, _, encoded_images = model.encoder.predict(images, verbose=0)
            reconstructed_images = model.decoder(encoded_images)

            fig = plt.figure(figsize=(1 * self.no_images, 4 * self.no_images))
            fig.suptitle(f"{name.upper()} at epoch {epoch + 1}", fontsize=self.text_size)
            fig.subplots_adjust(top=1.6)

            index = 0
            for i, image in enumerate(images):
                plt.subplot(self.no_images, 2, index + 1)
                plt.axis("off")
                if index == 0:
                    plt.title("Input", fontsize=self.text_size)
                plt.imshow(image / 2. + 0.5, interpolation="nearest")

                plt.subplot(self.no_images, 2, index + 2)
                plt.axis("off")
                if index == 0:
                    plt.title("Reconstructed", fontsize=self.text_size)
                plt.imshow(reconstructed_images[i] / 2. + 0.5, interpolation="nearest")

                index += 2

            fig.tight_layout()
            plt.savefig(os.sep.join([self.log_folder, "images", f"{name.upper()}_at_epoch_{epoch + 1:04d}.png"]))
            plt.close(fig)
