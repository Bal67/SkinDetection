import pandas as pd
import numpy as np
import boto3
from PIL import Image
from io import BytesIO
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score
from skimage.feature import canny
from skimage.color import rgb2gray
import joblib
import os
from concurrent.futures import ThreadPoolExecutor


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

# Function to preprocess a single image using Canny edge detector
def preprocess_image(img):
    gray_img = rgb2gray(np.array(img.resize((128, 128))))  # Resize for consistent shape
    edges = canny(gray_img)
    return edges.flatten()

# Function to extract features and labels for a batch of data
def extract_features_and_labels(df, bucket):
    X = []
    y = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        results = list(executor.map(lambda row: (row['augmented_image'], row['label'], download_image_from_s3(bucket, row['augmented_image'])), [row for _, row in df.iterrows()]))
    for img_key, label, img in results:
        if img is not None:
            features = preprocess_image(img)
            X.append(features)
            y.append(label)
    return np.array(X), np.array(y)

# Main function for training and evaluating the model
def main():
    s3_bucket = '540skinappbucket'
    data_file = '/content/drive/MyDrive/SCIN_Project/data/fitzpatrick17k_processed_augmented.csv'

    print("Loading data...")
    # Load data
    df = pd.read_csv(data_file)

    # Drop rows with None images
    df.dropna(subset=['augmented_image'], inplace=True)

    # Define label mappings
    label_encoder = LabelEncoder()
    df['label'] = label_encoder.fit_transform(df['label'])

    # Perform minimum sampling
    min_samples = df['label'].value_counts().min()
    balanced_df = df.groupby('label').apply(lambda x: x.sample(n=min_samples)).reset_index(drop=True)

    print("Splitting data...")
    # Split the data into train, validation, and test sets
    train_df, test_df = train_test_split(balanced_df, test_size=0.2, random_state=42)
    val_df, test_df = train_test_split(test_df, test_size=0.5, random_state=42)

    print("Extracting features...")
    # Extract features and labels
    X_train, y_train = extract_features_and_labels(train_df, s3_bucket)
    X_val, y_val = extract_features_and_labels(val_df, s3_bucket)
    X_test, y_test = extract_features_and_labels(test_df, s3_bucket)

    print(f"Training data shape: {X_train.shape}")
    print(f"Validation data shape: {X_val.shape}")
    print(f"Test data shape: {X_test.shape}")

    print("Training SVM classifier...")
    # Train an SVM classifier
    clf = SVC(kernel='linear')
    clf.fit(X_train, y_train)

    print("Evaluating the model on validation set...")
    # Evaluate the model on the validation set
    y_val_pred = clf.predict(X_val)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    val_precision = precision_score(y_val, y_val_pred, average='macro')
    val_recall = recall_score(y_val, y_val_pred, average='macro')

    print(f"Validation Accuracy: {val_accuracy:.4f}")
    print(f"Validation Precision: {val_precision:.4f}")
    print(f"Validation Recall: {val_recall:.4f}")

    print("Evaluating the model on test set...")
    # Evaluate the model on the test set
    y_test_pred = clf.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred, average='macro')
    test_recall = recall_score(y_test, y_test_pred, average='macro')

    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Precision: {test_precision:.4f}")
    print(f"Test Recall: {test_recall:.4f}")

    # Save the trained model
    model_path = '/content/drive/MyDrive/SCIN_Project/models/svm_model.pkl'
    joblib.dump(clf, model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    main()
