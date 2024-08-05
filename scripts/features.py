import pandas as pd
import numpy as np
import os
import boto3
from PIL import Image, ImageOps
from io import BytesIO
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from concurrent.futures import ThreadPoolExecutor

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

# Function to apply specific augmentations to an image
def augment_image(img):
    augmentations = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    ])
    return augmentations(img)

# Function to classify skin tone based on the fitzpatrick scale
def classify_skin_tone(fitzpatrick_scale):
    return 'dark' if fitzpatrick_scale > 3 else 'light'

# Function to visualize original and augmented images
def visualize_augmentations(bucket, img_key):
    img = download_image_from_s3(bucket, img_key)
    if img:
        augmented_images = [img, horizontal_flip(img), vertical_flip(img)]
        augmented_images += [inverse_color(img), augment_image(img)]

        fig, axes = plt.subplots(1, len(augmented_images), figsize=(20, 5))
        for i, aug_img in enumerate(augmented_images):
            axes[i].imshow(aug_img)
            axes[i].set_title('Original' if i == 0 else f'Augmentation {i}')
            axes[i].axis('off')
        plt.show()

# Function to save image to S3
def save_image_to_s3(img, bucket, key):
    buffer = BytesIO()
    img.save(buffer, format="JPEG")
    buffer.seek(0)
    s3_client.put_object(Bucket=bucket, Key=key, Body=buffer)

# Function to process and augment all images
def process_all_images(df, bucket):
    with ThreadPoolExecutor(max_workers=10) as executor:
        executor.map(lambda row: process_and_augment_image(row, bucket), [row for _, row in df.iterrows()])

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

# Function to count the number of images in the S3 bucket
def count_augimages_in_bucket(bucket):
    paginator = s3_client.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=bucket, Prefix='augmented_images/')
    augimage_count = 0
    for page in pages:
        if 'Contents' in page:
            image_count += len(page['Contents'])
    print(f"Total augmented images in bucket '{bucket}': {augimage_count}")
    return augimage_count

if __name__ == "__main__":
    s3_bucket = '540skinappbucket'
    data_file = '/content/drive/MyDrive/SCIN_Project/data/fitzpatrick17k_processed.csv'

    # Load data
    df = load_data(data_file)

    # Count images in the bucket
    count_images_in_bucket(s3_bucket)
    count_augimages_in_bucket(s3_bucket)

    # Visualize augmentations
    visualize_augmentations(s3_bucket, 'images/000000000000.jpg')
    