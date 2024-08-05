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
from torch import nn, optim
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from concurrent.futures import ThreadPoolExecutor

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

# Function to process a batch of images and extract features
def process_batch(batch, bucket, model):
    X_batch = []
    y_batch = []
    for row in batch:
        img_key = f"augmented_images/{row['md5hash']}.jpg"
        img = download_image_from_s3(bucket, img_key)
        if img is None:
            continue
        img_tensor = preprocess_image(img)
        features = extract_features(model, img_tensor)
        X_batch.append(features)
        y_batch.append(row['label'])
    return np.array(X_batch), np.array(y_batch)

# Function to process images and extract features in parallel
def process_images(df, bucket, model, batch_size=100):
    X = []
    y = []
    batches = [df[i:i + batch_size] for i in range(0, len(df), batch_size)]
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(process_batch, batch, bucket, model) for batch in batches]
        for future in futures:
            X_batch, y_batch = future.result()
            if X_batch.size > 0 and y_batch.size > 0:
                X.append(X_batch)
                y.append(y_batch)
    
    return np.vstack(X), np.concatenate(y)

# Function to train a fine-tuned ResNet model
def train_fine_tuned_resnet(train_loader, val_loader, num_classes, num_epochs=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(num_epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        model.eval()
        val_loss = 0.0
        val_correct = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                val_correct += torch.sum(preds == labels.data)
        
        val_loss /= len(val_loader.dataset)
        val_accuracy = val_correct.double() / len(val_loader.dataset)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {val_loss}, Accuracy: {val_accuracy}')
    
    return model

# Function to evaluate a model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    return accuracy, precision, recall

# Main function
def main():
    s3_bucket = '540skinappbucket'
    data_file = '/content/drive/MyDrive/SCIN_Project/data/fitzpatrick17k_processed.csv'
    
    df = load_data(data_file)
    X, y = process_images(df.to_dict('records'), s3_bucket, resnet_model)

    # Split data into train, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # Train non-fine-tuned logistic regression model
    non_fine_tuned_model = LogisticRegression(max_iter=1000)
    non_fine_tuned_model.fit(X_train, y_train)

    # Evaluate the non-fine-tuned model
    val_accuracy, val_precision, val_recall = evaluate_model(non_fine_tuned_model, X_val, y_val)
    print("Non-fine-tuned Validation Accuracy:", val_accuracy)
    print("Non-fine-tuned Validation Precision:", val_precision)
    print("Non-fine-tuned Validation Recall:", val_recall)

    test_accuracy, test_precision, test_recall = evaluate_model(non_fine_tuned_model, X_test, y_test)
    print("Non-fine-tuned Test Accuracy:", test_accuracy)
    print("Non-fine-tuned Test Precision:", test_precision)
    print("Non-fine-tuned Test Recall:", test_recall)

    # Save the non-fine-tuned model
    non_fine_tuned_model_path = '/content/drive/MyDrive/SCIN_Project/models/non_fine_tuned_logistic_regression.pkl'
    joblib.dump(non_fine_tuned_model, non_fine_tuned_model_path)
    print(f"Non-fine-tuned model saved to {non_fine_tuned_model_path}")

    # Create data loaders for fine-tuning
    train_dataset = torch.utils.data.TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train))
    val_dataset = torch.utils.data.TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val))
    test_dataset = torch.utils.data.TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Train fine-tuned ResNet model
    num_classes = len(np.unique(y_train))
    fine_tuned_model = train_fine_tuned_resnet(train_loader, val_loader, num_classes)

    # Save the fine-tuned model
    fine_tuned_model_path = '/content/drive/MyDrive/SCIN_Project/models/fine_tuned_resnet.pth'
    torch.save(fine_tuned_model.state_dict(), fine_tuned_model_path)
    print(f"Fine-tuned model saved to {fine_tuned_model_path}")

    # Evaluate the fine-tuned model
    fine_tuned_model.eval()
    y_val_pred = []
    y_val_true = []
    y_test_pred = []
    y_test_true = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fine_tuned_model = fine_tuned_model.to(device)
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = fine_tuned_model(images)
            _, preds = torch.max(outputs, 1)
            y_val_pred.extend(preds.cpu().numpy())
            y_val_true.extend(labels.cpu().numpy())

        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = fine_tuned_model(images)
            _, preds = torch.max(outputs, 1)
            y_test_pred.extend(preds.cpu().numpy())
            y_test_true.extend(labels.cpu().numpy())

    fine_tuned_val_accuracy = accuracy_score(y_val_true, y_val_pred)
    fine_tuned_val_precision = precision_score(y_val_true, y_val_pred, average='weighted')
    fine_tuned_val_recall = recall_score(y_val_true, y_val_pred, average='weighted')

    fine_tuned_test_accuracy = accuracy_score(y_test_true, y_test_pred)
    fine_tuned_test_precision = precision_score(y_test_true, y_test_pred, average='weighted')
    fine_tuned_test_recall = recall_score(y_test_true, y_test_pred, average='weighted')

    print("Fine-tuned Validation Accuracy:", fine_tuned_val_accuracy)
    print("Fine-tuned Validation Precision:", fine_tuned_val_precision)
    print("Fine-tuned Validation Recall:", fine_tuned_val_recall)

    print("Fine-tuned Test Accuracy:", fine_tuned_test_accuracy)
    print("Fine-tuned Test Precision:", fine_tuned_test_precision)
    print("Fine-tuned Test Recall:", fine_tuned_test_recall)

if __name__ == "__main__":
    main()
