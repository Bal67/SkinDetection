import pandas as pd
import numpy as np
import boto3
from PIL import Image, ImageOps
from io import BytesIO
import torch
import torchvision.transforms as transforms
from torchvision import models
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
import joblib

# Initialize S3 client
s3_client = boto3.client('s3',  region_name='us-east-1')

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

# Function to preprocess a single image for feature extraction
def preprocess_image(img):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return preprocess(img).unsqueeze(0)

# Function to apply augmentations to an image
def augment_image(img):
    augmentations = [
        img,
        ImageOps.invert(img.convert('RGB')).convert('RGB'),
        img.transpose(Image.FLIP_LEFT_RIGHT),
        img.transpose(Image.FLIP_TOP_BOTTOM),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2)(img)
    ]
    return augmentations

# Function to extract features from an image tensor using a pre-trained model
def extract_features(model, img_tensor):
    with torch.no_grad():
        features = model(img_tensor)
    return features.numpy().flatten()

# Function to process images, apply augmentations, and extract features
def process_images(df, bucket, model):
    X = []
    y = []
    for _, row in df.iterrows():
        img_key = f"images/{row['md5hash']}.jpg"
        img = download_image_from_s3(bucket, img_key)
        if img is None:
            continue
        
        # Apply augmentations
        augmented_images = augment_image(img)
        
        for aug_img in augmented_images:
            img_tensor = preprocess_image(aug_img)
            features = extract_features(model, img_tensor)
            X.append(features)
            y.append(row['label'])
    
    return np.array(X), np.array(y)

# Main function for feature extraction and model training
def main():
    s3_bucket = '540skinappbucket'
    data_file = '/content/drive/MyDrive/SCIN_Project/data/fitzpatrick17k_processed.csv'

    # Load data
    df = pd.read_csv(data_file)

    # Load pre-trained ResNet model
    resnet_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    resnet_model = torch.nn.Sequential(*(list(resnet_model.children())[:-1]))
    resnet_model.eval()

    # Process images and extract features
    X, y = process_images(df, s3_bucket, resnet_model)

    # Split the data into train, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # Train a Logistic Regression model
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)

    # Evaluate the model on the validation set
    y_val_pred = clf.predict(X_val)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    val_precision = precision_score(y_val, y_val_pred, average='macro')
    val_recall = recall_score(y_val, y_val_pred, average='macro')

    print(f"Validation Accuracy: {val_accuracy:.4f}")
    print(f"Validation Precision: {val_precision:.4f}")
    print(f"Validation Recall: {val_recall:.4f}")

    # Evaluate the model on the test set
    y_test_pred = clf.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred, average='macro')
    test_recall = recall_score(y_test, y_test_pred, average='macro')

    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Precision: {test_precision:.4f}")
    print(f"Test Recall: {test_recall:.4f}")

    # Save the trained model
    model_path = 'non_fine_tuned_model.pkl'
    joblib.dump(clf, model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    main()
