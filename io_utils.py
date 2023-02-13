import shutil
import io
import tensorflow as tf
from matplotlib import pyplot as plt

from configuration import *


def ensure_folder_structure(*folders):
    is_absolute_path = os.path.isabs(folders[0])
    provided_paths = []
    for path_part in folders:
        provided_paths.extend(path_part.split(os.sep))
    folder_path = os.getcwd() if not is_absolute_path else "/"

    for folder_name in provided_paths:
        folder_path = os.path.join(folder_path, folder_name)
        if not os.path.isdir(folder_path):
            os.mkdir(folder_path)


def delete_folder(path):
    shutil.rmtree(path, ignore_errors=True)


def plot_to_image(matplotlib_figure, channels=3):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    # Save the plot to a PNG in memory.
    buffer = io.BytesIO()
    plt.savefig(buffer, format="png")
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(matplotlib_figure)
    buffer.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buffer.getvalue(), channels=channels)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image
    

def seconds_to_human_readable(time):
    days = time // 86400         # (60 * 60 * 24)
    hours = time // 3600 % 24    # (60 * 60) % 24
    minutes = time // 60 % 60 
    seconds = time % 60
    
    time_string = ""
    if days > 0:
        time_string += f"{days:.0f} day{'s' if days > 1 else ''}, "
    if hours > 0 or days > 0:
        time_string += f"{hours:02.0f}h:"
    time_string += f"{minutes:02.0f}m:{seconds:02.0f}s"
    
    return time_string
