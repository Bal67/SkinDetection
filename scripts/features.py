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

# Helper function to process and augment a single image
def process_and_augment_image(row, bucket):
    img_key = f"images/{row['md5hash']}.jpg"
    img = download_image_from_s3(bucket, img_key)
    if img is None:
        return

    skin_tone = classify_skin_tone(row['fitzpatrick_scale'])
    augmented_images = [horizontal_flip(img), vertical_flip(img)]

    if skin_tone == 'dark':
        augmented_images += [inverse_color(img), augment_image(img)]

    for i, aug_img in enumerate(augmented_images):
        aug_key = f"augmented_images/{row['md5hash']}_aug_{i}.jpg"
        try:
            s3_client.head_object(Bucket=bucket, Key=aug_key)
            print(f"Image {aug_key} already exists. Skipping.")
        except s3_client.exceptions.ClientError:
            save_image_to_s3(aug_img, bucket, aug_key)
            print(f"Saved augmented image {aug_key}")

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

if __name__ == "__main__":
    s3_bucket = '540skinappbucket'
    data_file = '/content/drive/MyDrive/SCIN_Project/data/fitzpatrick17k_processed.csv'

    # Load data
    df = load_data(data_file)

    # Count images in the bucket
    count_images_in_bucket(s3_bucket)

    # Test augmentations on a single image
    test_row = df.iloc[0]
    visualize_augmentations(s3_bucket, f"images/{test_row['md5hash']}.jpg")

    # Process and augment all images
    process_all_images(df, s3_bucket)
