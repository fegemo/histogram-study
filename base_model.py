import os
from tensorflow import keras

import io_utils


class BaseModel(keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def save_model_description(self, folder_path):
        io_utils.ensure_folder_structure(folder_path)
        with open(os.sep.join([folder_path, "model-description.txt"]), "w") as fh:
            for model in self.models:
                model.summary(print_fn=lambda x: fh.write(x + "\n"))
                fh.write("\n"*3)

    @property
    def models(self):
        raise Exception("The 'models' property was not implemented in the subclass of BaseModel")
