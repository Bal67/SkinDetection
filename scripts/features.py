import pandas as pd
import numpy as np
import os
import boto3
from PIL import Image, ImageOps
from io import BytesIO
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
import torchvision.transforms as transforms

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

# Function to count the number of augmented images in the S3 bucket
def count_augimages_in_bucket(bucket):
    paginator = s3_client.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=bucket, Prefix='augmented_images/')
    image_count = 0
    for page in pages:
        if 'Contents' in page:
            image_count += len(page['Contents'])
    print(f"Total augmented images in bucket '{bucket}': {image_count}")
    return image_count

# Helper function to process a single row
def process_row(row, bucket):
    img_key = f"images/{row['md5hash']}.jpg"
    img = download_image_from_s3(bucket, img_key)
    if img is None:
        return

    augmented_images = [img, horizontal_flip(img), vertical_flip(img)]
    if classify_skin_tone(row['fitzpatrick_scale']) == 'dark':
        augmented_images += [inverse_color(img), augment_image(img)]

    # Save augmented images to S3 if they don't already exist
    for i, aug_img in enumerate(augmented_images):
        aug_img_key = f"augmented/{row['md5hash']}_aug_{i}.jpg"
        if not check_image_exists(bucket, aug_img_key):
            save_image_to_s3(bucket, aug_img_key, aug_img)

# Function to check if an image already exists in S3
def check_image_exists(bucket, key):
    try:
        s3_client.head_object(Bucket=bucket, Key=key)
        return True
    except:
        return False

# Function to save an image to S3
def save_image_to_s3(bucket, key, img):
    buffer = BytesIO()
    img.save(buffer, 'JPEG')
    buffer.seek(0)
    s3_client.put_object(Bucket=bucket, Key=key, Body=buffer)

# Function to apply augmentations to images and save them to S3
def augment_and_save_images(df, bucket):
    with ThreadPoolExecutor(max_workers=10) as executor:
        executor.map(lambda row: process_row(row, bucket), [row for _, row in df.iterrows()])

# Function to apply specific augmentations to an image
def augment_image(img):
    augmentations = transforms.Compose([
        transforms.RandomHorizontalFlip(),  # Randomly flip the image horizontally
        transforms.RandomVerticalFlip(),  # Randomly flip the image vertically
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),  # Randomly change brightness, contrast, saturation, and hue
    ])
    return augmentations(img)

# Function to visualize original and augmented images
def visualize_augmentations(bucket, img_key):
    img = download_image_from_s3(bucket, img_key)
    if img:
        augmented_images = [img, horizontal_flip(img), vertical_flip(img), inverse_color(img), augment_image(img)]

        fig, axes = plt.subplots(1, len(augmented_images), figsize=(20, 5))
        for i, aug_img in enumerate(augmented_images):
            axes[i].imshow(aug_img)
            axes[i].set_title('Original' if i == 0 else f'Augmentation {i}')
            axes[i].axis('off')
        plt.show()

if __name__ == "__main__":
    s3_bucket = '540skinappbucket'
    data_file = '/content/drive/MyDrive/SCIN_Project/data/fitzpatrick17k_processed.csv'

    # Initialize S3 client
    s3_client = boto3.client('s3')

    # Load data
    df = load_data(data_file)

    # Count images in the bucket
    count_augimages_in_bucket(s3_bucket)

    # Apply augmentations to all images and save them to S3
    augment_and_save_images(df, s3_bucket)

    # Visualize augmentations for a sample image
    test_img_key = 'images/ecacdf96f4a54f76834361a445194e0e.jpg'
    visualize_augmentations(s3_bucket, test_img_key)
