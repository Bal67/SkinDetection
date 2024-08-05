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

# Function to preprocess a single image
def preprocess_image(img):
    preprocess = transforms.Compose([
        transforms.Resize(256),  # Resize the image to 256x256 pixels
        transforms.CenterCrop(224),  # Crop the center 224x224 pixels
        transforms.ToTensor(),  # Convert the image to a PyTorch tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize the image
    ])
    return preprocess(img).unsqueeze(0)  # Add a batch dimension and return

# Function to apply specific augmentations to an image
def augment_image(img):
    augmentations = transforms.Compose([
        transforms.RandomHorizontalFlip(),  # Randomly flip the image horizontally
        transforms.RandomVerticalFlip(),  # Randomly flip the image vertically
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),  # Randomly change brightness, contrast, saturation, and hue
    ])
    return augmentations(img)

# Specific augmentations: inverse color, horizontal flip, vertical flip
def inverse_color(img):
    return ImageOps.invert(img.convert('RGB')).convert('RGB')

def horizontal_flip(img):
    return img.transpose(Image.FLIP_LEFT_RIGHT)

def vertical_flip(img):
    return img.transpose(Image.FLIP_TOP_BOTTOM)

# Function to download an image from S3
def download_image_from_s3(bucket, key):
    try:
        response = s3_client.get_object(Bucket=bucket, Key=key)  # Get the object from S3
        img_data = response['Body'].read()  # Read the image data
        img = Image.open(BytesIO(img_data))  # Open the image
        return img
    except s3_client.exceptions.NoSuchKey:
        return None
    except Exception as e:
        return None

# Function to classify skin tone based on the fitzpatrick scale
def classify_skin_tone(fitzpatrick_scale):
    if fitzpatrick_scale <= 3:
        return 'light'
    else:
        return 'dark'

# Helper function to process a single row
def process_row(row, bucket):
    img_key = f"images/{row['md5hash']}.jpg"  # S3 key for the image
    img = download_image_from_s3(bucket, img_key)  # Download the image
    if img is None:
        return None
    
    # Augmentations based on skin tone
    skin_tone = classify_skin_tone(row['fitzpatrick_scale'])
    if skin_tone == 'light':
        num_augmentations = 1  # 1 additional image for light skin
        augmentations = [inverse_color, augment_image]
    else:
        num_augmentations = 3  # 3 additional images for dark skin
        augmentations = [inverse_color, horizontal_flip, vertical_flip, augment_image]
    
    augmented_images = [augmentations[np.random.randint(len(augmentations))](img) for _ in range(num_augmentations)]
    return [img] + augmented_images

# Function to apply augmentations to images and visualize them
def augment_and_visualize_images(df, bucket):
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(process_row, row, bucket): index for index, row in df.iterrows()}
        for future in futures:
            result = future.result()
            if result:
                visualize_images(result)

# Function to visualize original and augmented images
def visualize_images(images):
    fig, axes = plt.subplots(1, len(images), figsize=(20, 5))
    for i, img in enumerate(images):
        axes[i].imshow(img)
        if i == 0:
            axes[i].set_title('Original Image')
        else:
            axes[i].set_title(f'Augmentation {i}')
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

if __name__ == "__main__":
    s3_bucket = '540skinappbucket'  # S3 bucket name
    data_file = '/content/drive/MyDrive/SCIN_Project/data/fitzpatrick17k_processed.csv'  # Path to the processed dataset CSV file

    # Initialize S3 client
    s3_client = boto3.client('s3')

    # Load data
    df = load_data(data_file)

    # Count images in the bucket
    count_images_in_bucket(s3_bucket)

    # Test image key
    test_img_key = df.iloc[0]['md5hash']  # Use the first image in the dataframe for testing
    img_key = f"images/{test_img_key}.jpg"

    # Visualize augmentations for the first image
    visualize_images(s3_bucket, img_key)

    # Apply augmentations to all images and visualize them
    augment_and_visualize_images(df, s3_bucket)
