import os
import datetime
import tensorflow as tf

from vae import CVAE, PreviewGeneratedImagesCallback
from dataset_utils import load_rgb_dataset
from configuration import OptionParser
import setup


options, parser = OptionParser().parse(True)
print("Running with options:", parser.get_description(", ", ":"))

datetime_string = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
config_params = f"lat-{options.latent_dim},crop-{options.crop_size},chan-{options.channels},down={options.downsamples}"
log_folder = f"{options.log_folder}{os.sep}{datetime_string},{options.description},{config_params}"
parser.save_configuration(log_folder)

if options.verbose:
    print("Tensorflow version:", tf.__version__)

    if tf.test.gpu_device_name():
        print("Default GPU: {}".format(tf.test.gpu_device_name()))
    else:
        print("Not using a GPU - it will take long!!")

# check if datasets need unzipping
setup.ensure_datasets(options.verbose)

# setting the seed
tf.random.set_seed(options.seed)
if options.verbose:
    print(f"SEEDed tensorflow with {options.seed}")

# loading the dataset according to the options
train_ds, test_ds = load_rgb_dataset(options.dataset, options)

# instantiating the model
model = CVAE(options)
model.save_model_description(log_folder)
if options.verbose:
    model.encoder.summary()
    model.decoder.summary()


model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=options.lr1))

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_folder, write_graph=False)
preview_callback = PreviewGeneratedImagesCallback(log_folder, 10, train_ds, test_ds)

print(f"Starting training for {options.epochs} epochs...")
model.fit(train_ds.batch(options.batch),
          epochs=options.epochs,
          shuffle=False,
          validation_data=test_ds.batch(options.batch),
          verbose=2,
          callbacks=[
    tensorboard_callback,
    preview_callback,
])

# generating resulting images from the latent space
# z_sample = tf.random.uniform(shape=(25, options.latent_dim)) * 2 - 1
# decoded = model.decoder.predict(z_sample)

# for i, image in enumerate(decoded):
#     plt.subplot(5, 5, i + 1)
#     plt.imshow(image/2.+0.5, interpolation="nearest")
#     plt.axis("off")
# plt.show()

# print(f"Saving the generator...")
# model.save_generator()


# python train_vae.py pixel-landscapes --batch 32 --latent-dim 128 --crop-size 32 --channels 3 --downsamples 2 --epochs 500 --log-folder output/vae/studying-layers --verbose
# testar aumento no histogram loss (20, 40, 60)
# testar redução do kl loss
