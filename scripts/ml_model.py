import pandas as pd
import numpy as np
import boto3
from PIL import Image
from io import BytesIO
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from torch.utils.data import DataLoader, Dataset
from concurrent.futures import ThreadPoolExecutor
from torchvision import models
import joblib

# Initialize S3 client
s3_client = boto3.client('s3', region_name='us-east-1')

# Function to download an image from S3
def download_image_from_s3(bucket, key):
    try:
        response = s3_client.get_object(Bucket=bucket, Key=key)
        img_data = response['Body'].read()
        img = Image.open(BytesIO(img_data)).convert('RGB')
        return img
    except Exception as e:
        print(f"Error downloading image {key}: {e}")
        return None

# Function to preprocess a single image for model training
def preprocess_image(img):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return preprocess(img)

# Dataset class to handle loading and transforming images
class SkinConditionDataset(Dataset):
    def __init__(self, df, bucket):
        self.df = df
        self.bucket = bucket

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_key = row['augmented_image']
        img = download_image_from_s3(self.bucket, img_key)
        if img is None:
            return None, row['label']
        img_tensor = preprocess_image(img)
        label = row['label']
        return img_tensor, label

def collate_fn(batch):
    batch = list(filter(lambda x: x[0] is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)

# Main function for training and evaluating the model
def main():
    s3_bucket = '540skinappbucket'
    data_file = '/content/drive/MyDrive/SCIN_Project/data/fitzpatrick17k_processed_augmented.csv'

    # Load data
    df = pd.read_csv(data_file)

    # Drop rows with None images
    df.dropna(subset=['augmented_image'], inplace=True)

    # Define label mappings
    label_mapping = {label: idx for idx, label in enumerate(df['label'].unique())}
    df['label'] = df['label'].map(label_mapping)

    # Split the data into train, validation, and test sets
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    val_df, test_df = train_test_split(test_df, test_size=0.5, random_state=42)

    # Create datasets
    train_dataset = SkinConditionDataset(train_df, s3_bucket)
    val_dataset = SkinConditionDataset(val_df, s3_bucket)
    test_dataset = SkinConditionDataset(test_df, s3_bucket)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4, collate_fn=collate_fn)

    # Load pre-trained ResNet18 model
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, len(label_mapping))
    model = model.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    for epoch in range(10):  # Number of epochs
        model.train()
        for images, labels in train_loader:
            images = images.cuda()
            labels = labels.cuda()
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Validation loop
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.cuda()
                labels = labels.cuda()
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        print(f"Epoch {epoch + 1}, Validation Loss: {val_loss / len(val_loader)}, Validation Accuracy: {val_correct / val_total}")

    # Test loop
    model.eval()
    test_correct = 0
    test_total = 0
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.cuda()
            labels = labels.cuda()
            outputs = model(images)
            _, predicted = outputs.max(1)
            test_total += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    test_accuracy = test_correct / test_total
    test_precision = precision_score(all_labels, all_preds, average='macro')
    test_recall = recall_score(all_labels, all_preds, average='macro')

    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Precision: {test_precision:.4f}")
    print(f"Test Recall: {test_recall:.4f}")

    # Save the trained model
    model_path = 'non_fine_tuned_model.pth'
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    main()
