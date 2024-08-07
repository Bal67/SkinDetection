import tensorflow as tf
import pandas as pd
import numpy as np
import boto3
from PIL import Image
from io import BytesIO
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from concurrent.futures import ThreadPoolExecutor
import multiprocessing

# Set up AWS credentials
os.environ['AWS_ACCESS_KEY_ID'] = 'AKIAUWSBZ5K5OCTEDKGS'
os.environ['AWS_SECRET_ACCESS_KEY'] = '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'
os.environ['AWS_REGION'] = 'us-east-1'

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

# Function to preprocess a single image
def preprocess_image(img):
    img = img.resize((128, 128))  # Smaller image size for faster processing
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    return img_array

# Function to extract features and labels for a batch of data
def extract_features_and_labels(df, bucket):
    X = []
    y = []
    with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
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

    # Compute class weights
    class_weights = compute_class_weight('balanced', classes=np.unique(balanced_df['label']), y=balanced_df['label'])
    class_weights = {i: class_weights[i] for i in range(len(class_weights))}

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

    print("Building the model...")
    # Load the MobileNetV2 model
    base_model = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
    base_model.trainable = False

    # Add custom layers on top
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(len(label_encoder.classes_), activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    print("Training the model...")
    # Train the model
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    
    history = model.fit(
        X_train, y_train,
        epochs=10,  # Increase the number of epochs for better training
        validation_data=(X_val, y_val),
        batch_size=64,  # Increase batch size for faster training
        class_weight=class_weights,
        callbacks=[early_stopping]
    )

    print("Evaluating the model...")
    # Evaluate the model on the test set
    test_loss, test_accuracy = model.evaluate(X_test, y_test, batch_size=64)
    print(f"Test Accuracy: {test_accuracy:.4f}")

    # Save the trained model
    model_path = '/content/drive/MyDrive/SCIN_Project/models/finetuned_mobilenetv2'
    model.save(model_path, save_format='tf')
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    main()
