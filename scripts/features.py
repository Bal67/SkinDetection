import pandas as pd
import numpy as np
import boto3
from PIL import Image, ImageOps
from io import BytesIO
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
import torchvision.transforms as transforms

# Initialize S3 client
s3_client = boto3.client('s3')

# Function to load the dataset from a CSV file
def load_data(filepath):
    return pd.read_csv(filepath)

# Function to download an image from S3
def download_image_from_s3(bucket, key):
    try:
        response = s3_client.get_object(Bucket=bucket, Key=key)
        img_data = response['Body'].read()
        img = Image.open(BytesIO(img_data))
        return img
    except Exception as e:
        print(f"Error downloading image {key}: {e}")
        return None

# Specific augmentations: inverse color, horizontal flip, vertical flip
def inverse_color(img):
    return ImageOps.invert(img.convert('RGB')).convert('RGB')

def horizontal_flip(img):
    return img.transpose(Image.FLIP_LEFT_RIGHT)

def vertical_flip(img):
    return img.transpose(Image.FLIP_TOP_BOTTOM)

# Function to classify skin tone based on the fitzpatrick scale
def classify_skin_tone(fitzpatrick_scale):
    return 'dark' if fitzpatrick_scale > 3 else 'light'

# Helper function to process a single row
def process_row(row, bucket):
    augmented_images = []
    img_key = f"images/{row['md5hash']}.jpg"
    img = download_image_from_s3(bucket, img_key)
    if img is None:
        return augmented_images
    
    augmented_images.append(img)  # Add the original image

    # Augmentations based on skin tone
    skin_tone = classify_skin_tone(row['fitzpatrick_scale'])
    augmented_images.append(horizontal_flip(img))
    augmented_images.append(vertical_flip(img))

    if skin_tone == 'dark':
        augmented_images.append(inverse_color(img))
        augmented_images.append(augment_image(img))

    return augmented_images

# Function to apply specific augmentations to an image
def augment_image(img):
    augmentations = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    ])
    return augmentations(img)

# Function to visualize original and augmented images
def visualize_images(images):
    fig, axes = plt.subplots(1, len(images), figsize=(20, 5))
    for i, img in enumerate(images):
        axes[i].imshow(img)
        axes[i].set_title('Original Image' if i == 0 else f'Augmentation {i}')
        axes[i].axis('off')
    plt.show()

# Function to count the number of images in the S3 bucket
def count_images_in_bucket(bucket):
    paginator = s3_client.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=bucket, Prefix='images/')
    image_count = 0
    for page in pages:
        if 'Contents' in page:
            image_count += len(page['Contents'])
    print(f"Total images in bucket '{bucket}': {image_count}")
    return image_count

# Function to extract and visualize features from images
def extract_and_visualize_features(df, bucket):
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(process_row, row, bucket) for index, row in df.iterrows()]
        for future in futures:
            result = future.result()

# Test function to process and visualize a single image
def test_augmentations_on_single_image(df, bucket):
    sample_row = df.iloc[0]
    augmented_images = process_row(sample_row, bucket)
    visualize_images(augmented_images)

if __name__ == "__main__":
    s3_bucket = '540skinappbucket'
    data_file = '/content/drive/MyDrive/SCIN_Project/data/fitzpatrick17k_processed.csv'

    # Load data
    df = load_data(data_file)

    # Count images in the bucket
    count_images_in_bucket(s3_bucket)

    # Extract and visualize features from all images
    extract_and_visualize_features(df, s3_bucket)

    # Test augmentations on a single image
    test_augmentations_on_single_image(df, s3_bucket)
