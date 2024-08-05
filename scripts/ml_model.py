import boto3
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from io import BytesIO
import torch
import torchvision.transforms as transforms
from torchvision import models
from torch import nn, optim
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
import joblib
import numpy as np

# Initialize S3 client
s3_client = boto3.client('s3')

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

# Function to process all images in the augmented_images folder and extract features
def process_all_images(bucket, folder, model):
    paginator = s3_client.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=bucket, Prefix=folder)
    
    X = []
    y = []
    for page in pages:
        if 'Contents' in page:
            for obj in page['Contents']:
                img_key = obj['Key']
                img = download_image_from_s3(bucket, img_key)
                if img is not None:
                    img_tensor = preprocess_image(img)
                    features = extract_features(model, img_tensor)
                    X.append(features)
                    label = img_key.split('/')[1].split('_')[0]  # Assuming label is the prefix before the first underscore
                    y.append(label)
    
    return np.array(X), np.array(y)

# Main function to train and evaluate models
def main():
    s3_bucket = '540skinappbucket'
    folder = 'augmented_images/'
    
    # Load pre-trained ResNet model
    resnet_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    resnet_model = torch.nn.Sequential(*(list(resnet_model.children())[:-1]))  # Remove the classification layer
    resnet_model.eval()
    
    # Process all images and extract features
    X, y = process_all_images(s3_bucket, folder, resnet_model)
    
    # Split the dataset into training, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    # Train the non-fine-tuned model (Logistic Regression)
    lr_model = LogisticRegression(max_iter=1000)
    lr_model.fit(X_train, y_train)
    
    # Evaluate the non-fine-tuned model
    y_val_pred = lr_model.predict(X_val)
    y_test_pred = lr_model.predict(X_test)
    
    val_accuracy = accuracy_score(y_val, y_val_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    val_precision = precision_score(y_val, y_val_pred, average='weighted')
    test_precision = precision_score(y_test, y_test_pred, average='weighted')
    val_recall = recall_score(y_val, y_val_pred, average='weighted')
    test_recall = recall_score(y_test, y_test_pred, average='weighted')
    
    print(f"Validation Accuracy: {val_accuracy}")
    print(f"Test Accuracy: {test_accuracy}")
    print(f"Validation Precision: {val_precision}")
    print(f"Test Precision: {test_precision}")
    print(f"Validation Recall: {val_recall}")
    print(f"Test Recall: {test_recall}")
    
    # Save the Logistic Regression model
    joblib.dump(lr_model, 'logistic_regression_model.pkl')

    # Train the fine-tuned model (ResNet50)
    fine_tuned_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    num_ftrs = fine_tuned_model.fc.in_features
    fine_tuned_model.fc = nn.Linear(num_ftrs, len(np.unique(y)))  # Assuming y contains the class labels
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(fine_tuned_model.parameters(), lr=0.001)
    
    # Move model to GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    fine_tuned_model.to(device)
    
    # Convert numpy arrays to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long).to(device)
    
    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        fine_tuned_model.train()
        
        optimizer.zero_grad()
        outputs = fine_tuned_model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
        
        fine_tuned_model.eval()
        val_outputs = fine_tuned_model(X_val_tensor)
        val_loss = criterion(val_outputs, y_val_tensor)
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}, Val Loss: {val_loss.item()}")
    
    # Save the fine-tuned model
    torch.save(fine_tuned_model.state_dict(), 'fine_tuned_resnet_model.pth')

if __name__ == "__main__":
    main()
