#Non-Fine Tuned Model
import os
import boto3
import joblib
import numpy as np
import pandas as pd
from PIL import Image
from io import BytesIO
import torch
import torchvision.transforms as transforms
from torchvision import models
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

# Initialize S3 client
s3_client = boto3.client('s3')

# Function to download an image from S3
def download_image_from_s3(bucket, key):
    try:
        response = s3_client.get_object(Bucket=bucket, Key=key)
        img_data = response['Body'].read()
        img = Image.open(BytesIO(img_data))
        return img
    except Exception as e:
        return None

# Function to preprocess an image for ResNet
def preprocess_image(img):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return preprocess(img).unsqueeze(0)

# Function to extract features using ResNet
def extract_features(model, img_tensor):
    with torch.no_grad():
        features = model(img_tensor)
    return features.numpy().flatten()

# Load pre-trained ResNet model
resnet_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
resnet_model = torch.nn.Sequential(*(list(resnet_model.children())[:-1]))  # Remove the classification layer
resnet_model.eval()

# Function to load and preprocess dataset
def load_data(filepath):
    df = pd.read_csv(filepath)
    return df

# Function to classify skin tone based on the fitzpatrick scale
def classify_skin_tone(fitzpatrick_scale):
    if fitzpatrick_scale <= 3:
        return 'light'
    else:
        return 'dark'

# Function to augment images
def augment_image(img):
    augmentations = [
        transforms.functional.hflip,
        transforms.functional.vflip,
    ]
    return [aug(img) for aug in augmentations]

# Function to process images and extract features
def process_images(df, bucket, model):
    X = []
    y = []
    for index, row in df.iterrows():
        img_key = f"images/{row['md5hash']}.jpg"
        img = download_image_from_s3(bucket, img_key)
        if img is None:
            continue
        img_tensor = preprocess_image(img)
        features = extract_features(model, img_tensor)
        X.append(features)
        y.append(row['label'])

        # Augmentations for dark skin tones
        if classify_skin_tone(row['fitzpatrick_scale']) == 'dark':
            augmented_images = augment_image(img)
            for aug_img in augmented_images:
                aug_img_tensor = preprocess_image(aug_img)
                aug_features = extract_features(model, aug_img_tensor)
                X.append(aug_features)
                y.append(row['label'])

    return np.array(X), np.array(y)

# Main function
def main():
    s3_bucket = '540skinappbucket'
    data_file = '/content/drive/MyDrive/SCIN_Project/data/fitzpatrick17k_processed.csv'
    
    df = load_data(data_file)
    X, y = process_images(df, s3_bucket, resnet_model)

    # Split data into train, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # Train logistic regression model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Evaluate the model
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)

    print("Validation Accuracy:", accuracy_score(y_val, y_val_pred))
    print("Validation Precision:", precision_score(y_val, y_val_pred, average='weighted'))
    print("Validation Recall:", recall_score(y_val, y_val_pred, average='weighted'))

    print("Test Accuracy:", accuracy_score(y_test, y_test_pred))
    print("Test Precision:", precision_score(y_test, y_test_pred, average='weighted'))
    print("Test Recall:", recall_score(y_test, y_test_pred, average='weighted'))

    # Save the model
    model_path = '/content/drive/MyDrive/SCIN_Project/models/non_fine_tuned_model.pkl'
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    main()
