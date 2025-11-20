import os
import zipfile

# Set environment variables for Kaggle config
os.environ['KAGGLE_CONFIG_DIR'] = os.path.join(os.path.dirname(__file__), '..')
os.environ['KAGGLE_CONFIG_FILE'] = 'kaggle.json'

from kaggle.api.kaggle_api_extended import KaggleApi

def download_and_extract_dataset():
    api = KaggleApi()
    api.authenticate()

    dataset = 'clmentbisaillon/fake-and-real-news-dataset'
    download_path = 'data/fake-and-real-news-dataset.zip'
    extract_path = 'data/'

    if not os.path.exists(download_path):
        print("Downloading dataset...")
        api.dataset_download_files(dataset, path='data', unzip=False)
    else:
        print("Dataset zip already exists.")

    # Extract the zip file
    if not os.path.exists(os.path.join(extract_path, 'Fake.csv')) or not os.path.exists(os.path.join(extract_path, 'True.csv')):
        print("Extracting dataset...")
        with zipfile.ZipFile(download_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        print("Extraction completed.")
    else:
        print("Dataset already extracted.")

if __name__ == "__main__":
    download_and_extract_dataset()
