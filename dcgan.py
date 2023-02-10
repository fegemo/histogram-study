import os

import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow import keras
from keras import layers

from base_model import BaseModel
from io_utils import ensure_folder_structure


class DCGAN(BaseModel):
    def __init__(self, options, **kwargs):
        super().__init__(**kwargs)
        self.options = options
        self.latent_dim = options.latent_dim
        self.d_steps = options.d_steps
        self.lambda_gp = options.lambda_gp
        self.discriminator = self.create_discriminator()
        self.generator = self.create_generator()
        self.d_loss_tracker = keras.metrics.Mean(name="d_loss")
        self.g_loss_tracker = keras.metrics.Mean(name="g_loss")
        self.gp_tracker = keras.metrics.Mean(name="gradient_penalty")
        self.loss_fn = None
        self.g_optimizer = None
        self.d_optimizer = None

    @property
    def models(self):
        return [self.discriminator, self.generator]

    def create_discriminator(self):
        size = self.options.crop_size
        channels = self.options.channels
        init = tf.random_normal_initializer(0., 0.02)

        input_image = keras.Input(shape=(size, size, channels))
        x = input_image

        filters = 128
        for i in range(self.options.downsamples):
            x = layers.Conv2D(filters, kernel_size=4, strides=2, padding="same", kernel_initializer=init)(x)
            x = layers.LeakyReLU()(x)
            if i > 0:
                x = layers.Dropout(0.5)(x)
            filters *= 2

        x = layers.Flatten()(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(1)(x)
        predicted_patches = x
        model = keras.Model(inputs=input_image, outputs=predicted_patches, name="discriminator")
        return model

    def create_generator(self):
        upsamples = self.options.downsamples
        channels = self.options.channels

        size = self.options.crop_size // (2 ** upsamples)
        init = tf.random_normal_initializer(0., 0.02)

        input_latent = keras.Input(shape=(self.latent_dim,))
        x = input_latent

        x = layers.Dense(size * size * 256, use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Reshape((size, size, 256))(x)

        filters = tf.cast(2 ** (6 + upsamples), tf.int32)
        for i in range(upsamples):
            x = layers.Conv2DTranspose(filters, kernel_size=4, strides=2, padding="same", use_bias=False,
                                       kernel_initializer=init)(x)
            x = layers.BatchNormalization()(x)
            x = layers.ReLU()(x)
            filters //= 2

        output_image = layers.Conv2D(channels, kernel_size=4, strides=1, padding="same", activation="tanh",
                                     kernel_initializer=init)(x)
        model = keras.Model(inputs=[input_latent], outputs=[output_image], name="generator")
        return model

    def compile(self, d_optimizer, g_optimizer):
        super().compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer

    @staticmethod
    def generator_loss(fake_predicted):
        return -tf.reduce_mean(fake_predicted)

    @staticmethod
    def discriminator_loss(real_predicted, fake_predicted):
        real_loss = tf.reduce_mean(real_predicted)
        fake_loss = tf.reduce_mean(fake_predicted)

        return fake_loss - real_loss

    def calculate_gradient_penalty(self, batch_size, real_images, fake_images):
        alpha = tf.random.uniform((batch_size, 1, 1, 1), minval=0, maxval=1)
        mixed_image = alpha * real_images + (1 - alpha) * fake_images

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(mixed_image)
            mixed_image_predicted = self.discriminator(mixed_image, training=True)

        gp_grads = gp_tape.gradient(mixed_image_predicted, mixed_image)
        gp_grad_norms = tf.sqrt(tf.reduce_sum(tf.square(gp_grads), axis=[1, 2, 3]))
        gradient_penalty = tf.reduce_mean(tf.square(gp_grad_norms - 1))

        return gradient_penalty

    @tf.function
    def train_step(self, real_images):
        batch_size = tf.shape(real_images)[0]

        # 1. train the discriminator first
        for i in range(self.d_steps):
            random_latent = tf.random.normal(shape=(batch_size, self.latent_dim))

            with tf.GradientTape() as tape:
                fake_images = self.generator(random_latent, training=True)

                fake_predicted = self.discriminator(fake_images, training=True)
                real_predicted = self.discriminator(real_images, training=True)

                d_cost = self.discriminator_loss(real_predicted, fake_predicted)
                gradient_penalty = self.calculate_gradient_penalty(batch_size, real_images, fake_images)
                d_loss = d_cost + gradient_penalty * self.lambda_gp

        d_gradients = tape.gradient(d_loss, self.discriminator.trainable_variables)
        self.d_optimizer.apply_gradients(zip(d_gradients, self.discriminator.trainable_variables))

        # 2. train the generator
        random_latent = tf.random.normal(shape=(batch_size, self.latent_dim))
        with tf.GradientTape() as tape:
            fake_images = self.generator(random_latent, training=True)
            fake_predicted = self.discriminator(fake_images, training=True)
            g_loss = self.generator_loss(fake_predicted)

        g_gradients = tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(g_gradients, self.generator.trainable_variables))

        # 3. update and return the metrics
        self.g_loss_tracker.update_state(g_loss)
        self.d_loss_tracker.update_state(d_loss)
        self.gp_tracker.update_state(gradient_penalty)

        return {m.name: m.result() for m in self.metrics}

    @property
    def metrics(self):
        return [self.d_loss_tracker, self.g_loss_tracker, self.gp_tracker]


class PreviewGeneratedImagesCallback(keras.callbacks.Callback):
    def __init__(self, log_folder, every_n_epochs, latent_dim, no_images=8):
        super().__init__()
        self.every_n_epochs = every_n_epochs
        self.log_folder = log_folder
        self.last_epoch_that_generated = -1
        self.last_epoch_watched = -1

        self.no_images = no_images
        self.random_latent = tf.random.normal(shape=(self.no_images, latent_dim))
        clipped_latent = tf.clip_by_value(self.random_latent, -1., 1.)
        barcode_height = tf.cast(latent_dim * 0.5, tf.int32)
        self.random_latent_barcode = tf.tile(clipped_latent[:, tf.newaxis, :], [1, barcode_height, 1])

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

        fake_images = model.generator(self.random_latent)
        fake_predicted = model.discriminator(fake_images)
        min_fake_predicted = tf.reduce_min(fake_predicted)
        fake_predicted = tf.tile(tf.keras.utils.normalize(fake_predicted, 0)[..., tf.newaxis], (1, 128, 128))

        fig = plt.figure(figsize=(2 * self.no_images, 5 * self.no_images))
        fig.suptitle(f"Images at epoch {epoch + 1}", fontsize=self.text_size)
        fig.subplots_adjust(top=1.6)

        index = 0
        for i, (image, predicted) in enumerate(zip(fake_images, fake_predicted)):
            plt.subplot(self.no_images, 3, index + 1)
            plt.axis("off")
            if index == 0:
                plt.title("Input latent", fontsize=self.text_size)
            plt.imshow(self.random_latent_barcode[i], interpolation="nearest", cmap="gray")

            plt.subplot(self.no_images, 3, index + 2)
            plt.axis("off")
            if index == 0:
                plt.title("Generated", fontsize=self.text_size)
            plt.imshow(image / 2. + 0.5, interpolation="nearest")

            plt.subplot(self.no_images, 3, index + 3)
            plt.axis("off")
            if index == 0:
                plt.title(f"Discriminated ({min_fake_predicted.numpy():.2f})", fontsize=self.text_size)
            plt.imshow(predicted, interpolation="nearest", cmap="gray")

            index += 3

        fig.tight_layout()
        plt.savefig(os.sep.join([self.log_folder, "images", f"generated_at_epoch_{epoch + 1:04d}.png"]))
        plt.close(fig)
