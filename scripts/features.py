import boto3
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from io import BytesIO
import torchvision.transforms as transforms
import numpy as np
import pandas as pd

# Initialize S3 client
s3_client = boto3.client('s3')

# Function to download an image from S3
def download_image_from_s3(bucket, key):
    try:
        response = s3_client.get_object(Bucket=bucket, Key=key)  # Get the object from S3
        img_data = response['Body'].read()  # Read the image data
        img = Image.open(BytesIO(img_data))  # Open the image
        return img
    except:
        return None

# Specific augmentations: inverse color, horizontal flip, vertical flip
def inverse_color(img):
    return ImageOps.invert(img.convert('RGB')).convert('RGB')

def horizontal_flip(img):
    return img.transpose(Image.FLIP_LEFT_RIGHT)

def vertical_flip(img):
    return img.transpose(Image.FLIP_TOP_BOTTOM)

# Function to apply additional augmentations to an image
def augment_image(img):
    augmentations = transforms.Compose([
        transforms.RandomHorizontalFlip(),  # Randomly flip the image horizontally
        transforms.RandomVerticalFlip(),  # Randomly flip the image vertically
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),  # Randomly change brightness, contrast, saturation, and hue
    ])
    return augmentations(img)

# Function to classify skin tone based on the Fitzpatrick scale
def classify_skin_tone(fitzpatrick_scale):
    return 'dark' if fitzpatrick_scale > 3 else 'light'

# Function to process a single image and apply augmentations
def process_image_and_augment(img, skin_tone):
    augmented_images = [img, horizontal_flip(img), vertical_flip(img)]
    
    if skin_tone == 'dark':
        augmented_images.append(inverse_color(img))
        augmented_images.append(augment_image(img))

    return augmented_images

# Function to visualize original and augmented images
def visualize_images(images):
    fig, axes = plt.subplots(1, len(images), figsize=(20, 5))
    for i, img in enumerate(images):
        axes[i].imshow(img)
        axes[i].set_title('Original Image' if i == 0 else f'Augmentation {i}')
        axes[i].axis('off')
    plt.show()

# Function to augment and visualize images from the dataset
def augment_and_visualize_images(df, bucket):
    for index, row in df.iterrows():
        img_key = f"images/{row['md5hash']}.jpg"  # S3 key for the image
        img = download_image_from_s3(bucket, img_key)
        if img is not None:
            skin_tone = classify_skin_tone(row['fitzpatrick_scale'])
            augmented_images = process_image_and_augment(img, skin_tone)
            visualize_images(augmented_images)

# Function to count the number of images in the S3 bucket
def count_images_in_bucket(bucket):
    paginator = s3_client.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=bucket, Prefix='images/')
    image_count = 0
    for page in pages:
        if 'Contents' in page:
            image_count += len(page['Contents'])
    return image_count

if __name__ == "__main__":
    s3_bucket = '540skinappbucket'  # S3 bucket name
    data_file = '/content/drive/MyDrive/SCIN_Project/data/fitzpatrick17k_processed.csv'  # Path to the processed dataset CSV file

    # Load data
    df = pd.read_csv(data_file)

    # Count images in the bucket
    image_count = count_images_in_bucket(s3_bucket)
    print(f"Total images in bucket '{s3_bucket}': {image_count}")

    # Apply augmentations to all images and visualize them
    augment_and_visualize_images(df, s3_bucket)
