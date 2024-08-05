import pandas as pd
import numpy as np
import os
import boto3
from PIL import Image, ImageOps
from io import BytesIO
import torch
import torchvision.transforms as transforms
from torchvision import models
from torch import nn, optim
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from concurrent.futures import ThreadPoolExecutor
import joblib

# Initialize S3 client
s3_client = boto3.client('s3')

# Function to load the dataset from a CSV file
def load_data(filepath):
    return pd.read_csv(filepath)

# Function to preprocess a single image for feature extraction
def preprocess_image(img):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return preprocess(img).unsqueeze(0)

# Function to download an image from S3
def download_image_from_s3(bucket, key):
    try:
        response = s3_client.get_object(Bucket=bucket, Key=key)
        img_data = response['Body'].read()
        img = Image.open(BytesIO(img_data))
        return img
    except Exception:
        return None

# Function to extract features from an image tensor using a pre-trained model
def extract_features(model, img_tensor):
    with torch.no_grad():
        features = model(img_tensor)
    return features.numpy().flatten()

# Helper function to process a single row
def process_row(row, bucket, model):
    features = []
    img_key_base = f"augmented_images/{row['md5hash']}"
    i = 0
    while True:
        img_key = f"{img_key_base}_aug_{i}.jpg"
        img = download_image_from_s3(bucket, img_key)
        if img is None:
            break
        img_tensor = preprocess_image(img)
        features.append(extract_features(model, img_tensor))
        i += 1
    return features

# Function to process images and extract features
def process_images(records, bucket, model):
    X = []
    y = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(process_row, row, bucket, model): index for index, row in enumerate(records)}
        for future in futures:
            result = future.result()
            if result:
                X.extend(result)
                y.extend([records[futures[future]]['label']] * len(result))
    return np.vstack(X), np.concatenate(y)

# Function to build and evaluate a logistic regression model
def build_and_evaluate_model(X_train, y_train, X_val, y_val, X_test, y_test):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)

    print("Training Accuracy: ", accuracy_score(y_train, y_train_pred))
    print("Validation Accuracy: ", accuracy_score(y_val, y_val_pred))
    print("Test Accuracy: ", accuracy_score(y_test, y_test_pred))
    print("Precision: ", precision_score(y_test, y_test_pred, average='weighted'))
    print("Recall: ", recall_score(y_test, y_test_pred, average='weighted'))

    return model

# Function to build and fine-tune a model
def fine_tune_model(X_train, y_train, X_val, y_val, X_test, y_test):
    class FineTuneModel(nn.Module):
        def __init__(self):
            super(FineTuneModel, self).__init__()
            self.resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            num_ftrs = self.resnet.fc.in_features
            self.resnet.fc = nn.Linear(num_ftrs, len(set(y_train)))

        def forward(self, x):
            return self.resnet(x)

    model = FineTuneModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Convert numpy arrays to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    # Create data loaders
    train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = torch.utils.data.TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = torch.utils.data.TensorDataset(X_test_tensor, y_test_tensor)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Training loop
    for epoch in range(10):  # Number of epochs can be adjusted
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}")

        # Validation loop
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print(f"Validation Loss: {val_loss/len(val_loader)}, Accuracy: {100 * correct / total}")

    # Test loop
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"Test Accuracy: {100 * correct / total}")

    return model

def main():
    s3_bucket = '540skinappbucket'
    data_file = '/content/drive/MyDrive/SCIN_Project/data/fitzpatrick17k_processed.csv'

    # Initialize S3 client
    s3_client = boto3.client('s3')

    # Load data
    df = load_data(data_file)

    # Load pre-trained model
    resnet_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    resnet_model = torch.nn.Sequential(*(list(resnet_model.children())[:-1]))  # Remove the classification layer
    resnet_model.eval()  # Set the model to evaluation mode

    # Process images and extract features
    X, y = process_images(df.to_dict('records'), s3_bucket, resnet_model)

    # Split dataset into train, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # Build and evaluate non-fine-tuned model
    print("Non-fine-tuned model evaluation:")
    non_fine_tuned_model = build_and_evaluate_model(X_train, y_train, X_val, y_val, X_test, y_test)
    joblib.dump(non_fine_tuned_model, 'non_fine_tuned_model.pkl')

    # Build and evaluate fine-tuned model
    print("Fine-tuned model evaluation:")
    fine_tuned_model = fine_tune_model(X_train, y_train, X_val, y_val, X_test, y_test)
    torch.save(fine_tuned_model.state_dict(), 'fine_tuned_model.pth')

if __name__ == "__main__":
    main()
