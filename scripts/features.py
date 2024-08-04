import pandas as pd
import numpy as np
import os
import boto3
from PIL import Image, ImageOps
from io import BytesIO
import torch
import torchvision.transforms as transforms
from torchvision import models
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed

# Function to load the dataset from a CSV file
def load_data(filepath):
    return pd.read_csv(filepath)

# Function to preprocess a single image for feature extraction
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

# Function to extract features from an image tensor using a pre-trained model
def extract_features(model, img_tensor):
    with torch.no_grad():  # Disable gradient calculation
        features = model(img_tensor)  # Extract features
    return features.numpy().flatten()  # Flatten the features and convert to a numpy array

# Function to download an image from S3
def download_image_from_s3(bucket, key):
    try:
        response = s3_client.get_object(Bucket=bucket, Key=key)  # Get the object from S3
        img_data = response['Body'].read()  # Read the image data
        img = Image.open(BytesIO(img_data))  # Open the image
        return img
    except s3_client.exceptions.NoSuchKey:
        print(f"Image with key {key} does not exist.")
        return None
    except Exception as e:
        return None

# Function to classify skin tone based on the fitzpatrick scale
def classify_skin_tone(fitzpatrick_scale):
    if fitzpatrick_scale <= 3:
        return 'light'
    else:
        return 'dark'

# Function to remove rows with missing images
def remove_rows_with_missing_images(df, bucket):
    def image_exists(row):
        key = f"images/{row['md5hash']}.jpg"
        try:
            s3_client.head_object(Bucket=bucket, Key=key)
            return True
        except s3_client.exceptions.NoSuchKey:
            return False
    return df[df.apply(image_exists, axis=1)]

# Function to check if features are already processed
def features_already_processed(feature_dir, md5hash):
    return os.path.exists(os.path.join(feature_dir, f'{md5hash}.npy'))

# Function to extract features from images and save them to individual numpy files
def extract_features_from_images(df, bucket, model, feature_dir):
    os.makedirs(feature_dir, exist_ok=True)  # Ensure the feature directory exists
    
    # Remove rows with missing images
    df = remove_rows_with_missing_images(df, bucket)
    
    features_list = []  # List to store features
    missing_images = 0  # Counter for missing images

    def process_row(row):
        img_key = f"images/{row['md5hash']}.jpg"  # S3 key for the image

        if features_already_processed(feature_dir, row['md5hash']):
            print(f"Features for {img_key} already processed. Skipping.")
            return None
        
        img = download_image_from_s3(bucket, img_key)  # Download the image
        if img is None:
            return None
        
        # Original image
        img_tensor = preprocess_image(img)
        features = extract_features(model, img_tensor)
        np.save(os.path.join(feature_dir, f'{row["md5hash"]}.npy'), features)
        
        # Augmentations based on skin tone
        skin_tone = classify_skin_tone(row['fitzpatrick_scale'])
        if skin_tone == 'light':
            num_augmentations = 1  # 1 additional image for light skin
            augmentations = [inverse_color, augment_image]
        else:
            num_augmentations = 3  # 3 additional images for dark skin
            augmentations = [inverse_color, horizontal_flip, vertical_flip, augment_image]
        
        augmented_images = [augmentations[np.random.randint(len(augmentations))](img) for _ in range(num_augmentations)]
        for aug_img in augmented_images:
            augmented_tensor = preprocess_image(aug_img)
            augmented_features = extract_features(model, augmented_tensor)
            np.save(os.path.join(feature_dir, f'{row["md5hash"]}_aug_{np.random.randint(1000)}.npy'), augmented_features)
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(process_row, row) for _, row in df.iterrows()]
        for future in as_completed(futures):
            result = future.result()
            if result is None:
                missing_images += 1

if __name__ == "__main__":
    s3_bucket = '540skinappbucket'  # S3 bucket name
    data_file = '/content/drive/MyDrive/SCIN_Project/data/fitzpatrick17k_processed.csv'  # Path to the processed dataset CSV file
    feature_dir = '/content/drive/MyDrive/SCIN_Project/data/features'  # Directory to save the features

    # Initialize S3 client
    s3_client = boto3.client('s3')

    # Load data
    df = load_data(data_file)

    # Load pre-trained model
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model = torch.nn.Sequential(*(list(model.children())[:-1]))  # Remove the classification layer
    model.eval()  # Set the model to evaluation mode

    # Extract features
    try:
        extract_features_from_images(df, s3_bucket, model, feature_dir)
    except Exception as e:
        print(f"An error occurred during feature extraction: {e}")

    # Test section
    print("Running tests...")

    # Test image key
    test_img_key = df.iloc[0]['md5hash']  # Use the first image in the dataframe for testing
    img_key = f"images/{test_img_key}.jpg"

    # Download the image
    img = download_image_from_s3(s3_bucket, img_key)
    if img:
        print(f"Downloaded image size: {img.size}")

        # Apply augmentations
        augmented_images = [img, inverse_color(img), horizontal_flip(img), vertical_flip(img)] + [augment_image(img)]
        
        # Display the original and augmented images
        fig, axes = plt.subplots(1, len(augmented_images), figsize=(20, 5))
        for i, aug_img in enumerate(augmented_images):
            axes[i].imshow(aug_img)
            if i == 0:
                axes[i].set_title('Original Image')
            else:
                axes[i].set_title(f'Augmentation {i}')
        plt.show()

        # Preprocess the image
        img_tensor = preprocess_image(img)
        print(f"Preprocessed image tensor shape: {img_tensor.shape}")

        # Extract features
        features = extract_features(model, img_tensor)
        print(f"Extracted features shape: {features.shape}")
        print(f"Extracted features type: {type(features)}")
        print(f"Extracted features: {features[:10]}")  # Print first 10 features for inspection

    print("Tests completed.")
