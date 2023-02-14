import requests

from configuration import *
import os
import zipfile
from tqdm import tqdm


def ensure_datasets(verbose=True):
    for name, folder, mask in zip(DATASET_NAMES, DATA_FOLDERS, DATASET_MASK):
        if mask != 1:
            continue

        datasets_folder = folder[:(folder.index(name)-1)]
        # (a) a folder with the name of the dataset must exist; if not, (b) the .zip; if not, (c) the .txt with the url
        folder_exists = os.path.exists(folder)
        zip_exists = os.path.exists(folder + ".zip")
        txt_exists = os.path.exists(folder + ".txt")

        if not folder_exists:
            if zip_exists:
                unzip_dataset(folder + ".zip", datasets_folder, verbose)
            elif txt_exists:
                zip_url = read_zip_path(folder + ".txt", verbose)
                download_zip(zip_url, folder + ".zip", verbose)
                unzip_dataset(folder + ".zip", datasets_folder, verbose)

        else:
            if verbose:
                print(f"Dataset {name} found!")


def read_zip_path(txt_file_path, verbose=False):
    if verbose:
        print(f"Will read {txt_file_path} to get the url to download")
    with open(txt_file_path, "r") as txt:
        zip_url = txt.readline()
    return zip_url


def download_zip(zip_url, zip_file_path, verbose=False):
    requests.packages.urllib3.disable_warnings()
    if verbose:
        print(f"Will download {zip_file_path} from the Internet: {zip_url}")
    response = requests.get(zip_url, stream=True, verify=False)
    with open(zip_file_path, "wb") as fd:
        for chunk in response.iter_content(chunk_size=128):
            fd.write(chunk)


def unzip_dataset(zip_file_path, output_folder, verbose=False):
    if verbose:
        print(f"Will unzip {zip_file_path} dataset to {output_folder}")
    with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
        for member in tqdm(zip_ref.infolist(), desc=f"Extracting {zip_file_path}"):
            zip_ref.extract(member, output_folder)
    if verbose:
        print(f"Extracted {output_folder} - dataset ready")


if __name__ == "__main__":
    ensure_datasets()
