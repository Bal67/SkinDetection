import pandas as pd
import os
import requests
from PIL import Image
from io import BytesIO
import boto3
from concurrent.futures import ThreadPoolExecutor, as_completed

# Function to load the dataset from a CSV file
def load_data_from_github(url):
    return pd.read_csv(url)

# Function to preprocess the dataset
def preprocess_data(df):
    df = df.dropna(subset=['url'])  # Drop rows without URLs
    df = df.dropna(subset=['label'])  # Drop rows without labels
    df = df[~df['url'].str.contains('dermaamin.com')]  # Remove rows with URLs from dermaamin.com
    return df

# Function to check if an image exists in S3
def image_exists_in_s3(bucket, key):
    try:
        s3_client.head_object(Bucket=bucket, Key=key)
        return True
    except s3_client.exceptions.ClientError as e:
        if e.response['Error']['Code'] == '404':
            return False
        else:
            print(f"Error checking if {key} exists in S3: {e}")
            return False

# Function to download an image from a URL
def download_image(url):
    try:
        response = requests.get(url, timeout=10)
        img = Image.open(BytesIO(response.content))
        return img
    except Exception:
        return None

# Function to upload an image to S3
def upload_to_s3(img, bucket, key):
    buffer = BytesIO()
    img.save(buffer, 'JPEG')
    buffer.seek(0)
    s3_client.upload_fileobj(buffer, bucket, key)

# Function to download and upload an image if it doesn't exist in S3
def download_and_upload_image(row, bucket):
    key = f"images/{row['md5hash']}.jpg"
    if image_exists_in_s3(bucket, key):
        print(f"Image {key} already exists in S3. Skipping download.")
        return True
    img = download_image(row['url'])
    if img:
        upload_to_s3(img, bucket, key)
        return True
    return False

# Function to process a batch of images
def process_batch(batch_df, bucket):
    successful_downloads = 0
    failed_downloads = 0

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = []
        for _, row in batch_df.iterrows():
            futures.append(executor.submit(download_and_upload_image, row, bucket))

        for future in as_completed(futures):
            if future.result():
                successful_downloads += 1
            else:
                failed_downloads += 1

    return successful_downloads, failed_downloads

# Function to download images in batches
def download_images_in_batches(df, bucket, batch_size=500):
    total_successful_downloads = 0
    total_failed_downloads = 0

    print(f"Total number of rows in the dataset: {len(df)}")

    for i in range(0, len(df), batch_size):
        batch_df = df.iloc[i:i+batch_size]
        successful_downloads, failed_downloads = process_batch(batch_df, bucket)
        total_successful_downloads += successful_downloads
        total_failed_downloads += failed_downloads



# Function to save the dataset to a file
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
    download_images_in_batches(df, s3_bucket, batch_size=500)
    save_data(df, 'fitzpatrick17k+_processed.csv', '/content/drive/MyDrive/SCIN_Project/data')
