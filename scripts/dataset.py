import pandas as pd
import os
import requests
from PIL import Image
from io import BytesIO
import boto3
from concurrent.futures import ThreadPoolExecutor, as_completed

def load_data_from_github(url):
    return pd.read_csv(url)

def preprocess_data(df):
    df = df.dropna(subset=['url'])  # Drop rows without URLs
    df = df.dropna(subset=['label'])  # Drop rows without labels
    df = df[~df['url'].str.contains('dermaamin.com')]  # Remove rows with URLs from dermaamin.com
    return df

def download_image(url):
    try:
        response = requests.get(url, timeout=10)
        img = Image.open(BytesIO(response.content))
        return img
    except Exception:
        return None

def upload_to_s3(img, bucket, key):
    buffer = BytesIO()
    img.save(buffer, 'JPEG')
    buffer.seek(0)
    s3_client.upload_fileobj(buffer, bucket, key)

def download_and_upload_image(row, bucket):
    img = download_image(row['url'])
    if img:
        key = f"images/{row['md5hash']}.jpg"
        upload_to_s3(img, bucket, key)
        return True
    return False

def process_batch(batch_df, bucket):
    successful_downloads = 0
    failed_downloads = 0

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = []
        for index, row in batch_df.iterrows():
            futures.append(executor.submit(download_and_upload_image, row, bucket))

        for future in as_completed(futures):
            if future.result():
                successful_downloads += 1
            else:
                failed_downloads += 1

    return successful_downloads, failed_downloads

def download_images_in_batches(df, bucket, batch_size=500):
    total_successful_downloads = 0
    total_failed_downloads = 0

    print(f"Total number of rows in the dataset: {len(df)}")

    for i in range(0, len(df), batch_size):
        batch_df = df.iloc[i:i+batch_size]
        successful_downloads, failed_downloads = process_batch(batch_df, bucket)
        total_successful_downloads += successful_downloads
        total_failed_downloads += failed_downloads

    print(f"Successfully downloaded and uploaded {total_successful_downloads} images.")
    print(f"Failed to download and upload {total_failed_downloads} images.")

def save_data(df, filename, directory):
    os.makedirs(directory, exist_ok=True)
    df.to_csv(os.path.join(directory, filename), index=False)

if __name__ == "__main__":
    github_url = 'https://raw.githubusercontent.com/Bal67/SkinDetection/main/data/fitzpatrick17k.csv'
    s3_bucket = '540skinappbucket'

    # Initialize S3 client
    s3_client = boto3.client('s3')

    df = load_data_from_github(github_url)
    df = preprocess_data(df)
    download_images_in_batches(df, s3_bucket, batch_size=500)
    save_data(df, 'fitzpatrick17k.csv', 'data/processed')
