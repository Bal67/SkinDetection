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
    except Exception as e:
        print(f"Failed to download image from {url}: {e}")
        return None

def upload_to_s3(img, bucket, key):
    buffer = BytesIO()
    img.save(buffer, 'JPEG')
    buffer.seek(0)
    s3_client.upload_fileobj(buffer, bucket, key)
    print(f"Uploaded {key} to S3")

def download_and_upload_image(row, bucket):
    img = download_image(row['url'])
    if img:
        key = f"images/{row['md5hash']}.jpg"
        upload_to_s3(img, bucket, key)
        return True
    return False

def filter_rows_with_unavailable_images(df, bucket):
    available_rows = []

    def check_image_availability(row):
        img = download_image(row['url'])
        if img:
            key = f"images/{row['md5hash']}.jpg"
            upload_to_s3(img, bucket, key)
            available_rows.append(row)

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(check_image_availability, row) for _, row in df.iterrows()]
        for future in as_completed(futures):
            future.result()

    return pd.DataFrame(available_rows)

def save_data(df, filename, directory):
    os.makedirs(directory, exist_ok=True)
    df.to_csv(os.path.join(directory, filename), index=False)

if __name__ == "__main__":
    github_url = 'https://raw.githubusercontent.com/Bal67/SkinDetection/main/data/fitzpatrick17k.csv'
    s3_bucket = '540skinappbucket'

    # Get AWS credentials from environment variables
    AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
    AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
    AWS_REGION = os.getenv('AWS_REGION')

    # Initialize S3 client with credentials
    s3_client = boto3.client('s3', 
                             aws_access_key_id=AWS_ACCESS_KEY_ID, 
                             aws_secret_access_key=AWS_SECRET_ACCESS_KEY, 
                             region_name=AWS_REGION)

    df = load_data_from_github(github_url)
    df = preprocess_data(df)
    
    # Filter rows where images cannot be downloaded
    df = filter_rows_with_unavailable_images(df, s3_bucket)
    
    save_data(df, 'fitzpatrick17k_processed.csv', '/content/drive/MyDrive/SCIN_Project/data')
