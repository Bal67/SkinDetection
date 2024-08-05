import pandas as pd
import numpy as np
import os
import boto3
from PIL import Image, ImageOps
from io import BytesIO
import torch
import torchvision.transforms as transforms
from torchvision import models
from concurrent.futures import ThreadPoolExecutor
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
    except s3_client.exceptions.NoSuchKey:
        print(f"Image with key {key} does not exist.")
        return None
    except Exception as e:
        print(f"An error occurred while downloading the image: {e}")
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

# Function to apply specific augmentations to an image
def augment_image(img):
    augmentations = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
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
    with torch.no_grad():
        features = model(img_tensor)
    return features.numpy().flatten()

# Function to classify skin tone based on the fitzpatrick scale
def classify_skin_tone(fitzpatrick_scale):
    return 'dark' if fitzpatrick_scale > 3 else 'light'

# Helper function to process a single image
def process_image(row, bucket, model):
    img_key = f"augmented_images/{row['md5hash']}.jpg"
    img = download_image_from_s3(bucket, img_key)
    if img is None:
        return None, None

    img_tensor = preprocess_image(img)
    features = extract_features(model, img_tensor)
    
    # Augmentations
    augmented_features = []
    if classify_skin_tone(row['fitzpatrick_scale']) == 'dark':
        augmentations = [horizontal_flip, vertical_flip, inverse_color]
        for aug in augmentations:
            aug_img = aug(img)
            aug_img_tensor = preprocess_image(aug_img)
            augmented_features.append(extract_features(model, aug_img_tensor))
    return [features] + augmented_features, row['label']

# Function to process images in parallel
def process_images(df, bucket, model):
    X, y = [], []
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(process_image, row, bucket, model) for row in df]
        for future in futures:
            result, label = future.result()
            if result is not None:
                X.extend(result)
                y.extend([label] * len(result))
    if not X or not y:
        raise ValueError("No images were processed. Please check your dataset and S3 bucket.")
    return np.vstack(X), np.array(y)

# Function to load the dataset from a CSV file
def load_data(filepath):
    return pd.read_csv(filepath)

# Function to train and evaluate a logistic regression model
def train_evaluate_model(X_train, y_train, X_val, y_val, X_test, y_test):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Evaluation on validation set
    y_val_pred = model.predict(X_val)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    val_precision = precision_score(y_val, y_val_pred, average='weighted')
    val_recall = recall_score(y_val, y_val_pred, average='weighted')

    # Evaluation on test set
    y_test_pred = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred, average='weighted')
    test_recall = recall_score(y_test, y_test_pred, average='weighted')

    return {
        "val_accuracy": val_accuracy,
        "val_precision": val_precision,
        "val_recall": val_recall,
        "test_accuracy": test_accuracy,
        "test_precision": test_precision,
        "test_recall": test_recall,
    }

# Main function
def main():
    s3_bucket = '540skinappbucket'
    data_file = '/content/drive/MyDrive/SCIN_Project/data/fitzpatrick17k_processed.csv'

    # Load data
    df = load_data(data_file)

    # Load pre-trained ResNet model
    resnet_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    resnet_model = torch.nn.Sequential(*(list(resnet_model.children())[:-1]))
    resnet_model.eval()

    # Process images and extract features
    X, y = process_images(df.to_dict('records'), s3_bucket, resnet_model)

    # Split dataset
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # Train and evaluate non-fine-tuned model
    metrics_non_fine_tuned = train_evaluate_model(X_train, y_train, X_val, y_val, X_test, y_test)
    print("Non-Fine-Tuned Model Evaluation:")
    print(metrics_non_fine_tuned)

    # Train and evaluate fine-tuned model
    fine_tuned_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    num_ftrs = fine_tuned_model.fc.in_features
    fine_tuned_model.fc = torch.nn.Linear(num_ftrs, len(set(y)))  # Adjust the final layer
    fine_tuned_model = fine_tuned_model.to(device='cuda' if torch.cuda.is_available() else 'cpu')

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(fine_tuned_model.parameters(), lr=0.001, momentum=0.9)
    dataloaders = {
        'train': torch.utils.data.DataLoader(
            [(x, y) for x, y in zip(X_train, y_train)],
            batch_size=32,
            shuffle=True
        ),
        'val': torch.utils.data.DataLoader(
            [(x, y) for x, y in zip(X_val, y_val)],
            batch_size=32,
            shuffle=False
        )
    }

    for epoch in range(10):  # Train for 10 epochs
        fine_tuned_model.train()
        for inputs, labels in dataloaders['train']:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = fine_tuned_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    
    fine_tuned_model.eval()
    y_val_true, y_val_pred = [], []
    for inputs, labels in dataloaders['val']:
        inputs = inputs.to(device)
        outputs = fine_tuned_model(inputs)
        _, preds = torch.max(outputs, 1)
        y_val_true.extend(labels.tolist())
        y_val_pred.extend(preds.tolist())

    fine_tuned_val_accuracy = accuracy_score(y_val_true, y_val_pred)
    fine_tuned_val_precision = precision_score(y_val_true, y_val_pred, average='weighted')
    fine_tuned_val_recall = recall_score(y_val_true, y_val_pred, average='weighted')

    y_test_true, y_test_pred = [], []
    for inputs, labels in dataloaders['test']:
        inputs = inputs.to(device)
        outputs = fine_tuned_model(inputs)
        _, preds = torch.max(outputs, 1)
        y_test_true.extend(labels.tolist())
        y_test_pred.extend(preds.tolist())

    fine_tuned_test_accuracy = accuracy_score(y_test_true, y_test_pred)
    fine_tuned_test_precision = precision_score(y_test_true, y_test_pred, average='weighted')
    fine_tuned_test_recall = recall_score(y_test_true, y_test_pred, average='weighted')

    print("Fine-tuned Model Evaluation:")
    print({
        "val_accuracy": fine_tuned_val_accuracy,
        "val_precision": fine_tuned_val_precision,
        "val_recall": fine_tuned_val_recall,
        "test_accuracy": fine_tuned_test_accuracy,
        "test_precision": fine_tuned_test_precision,
        "test_recall": fine_tuned_test_recall,
    })

    # Save the fine-tuned model
    model_save_path = '/content/drive/MyDrive/SCIN_Project/models/fine_tuned_resnet50.pth'
    torch.save(fine_tuned_model.state_dict(), model_save_path)
    print(f"Fine-tuned model saved at {model_save_path}")

if __name__ == "__main__":
    main()
