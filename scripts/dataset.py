import pandas as pd
import os
from sklearn.model_selection import train_test_split
import requests
from PIL import Image
from io import BytesIO
import zipfile

def load_data_from_github(url):
    return pd.read_csv(url)

def preprocess_data(df):
    df = df.dropna(subset=['url'])  # Drop rows without URLs
    return df

def download_image(url, save_path):
    try:
        response = requests.get(url)
        img = Image.open(BytesIO(response.content))
        img.save(save_path)
        return True
    except Exception as e:
        print(f"Failed to download {url}: {e}")
        return False

def download_images(df, batch_size=100):
    os.makedirs('data/raw/images', exist_ok=True)
    successful_downloads = 0
    failed_downloads = 0
    for i in range(0, len(df), batch_size):
        batch_df = df.iloc[i:i+batch_size]
        batch_dir = f'data/raw/images/batch_{i//batch_size + 1}'
        os.makedirs(batch_dir, exist_ok=True)
        for index, row in batch_df.iterrows():
            save_path = os.path.join(batch_dir, f"{row['md5hash']}.jpg")
            if download_image(row['url'], save_path):
                successful_downloads += 1
            else:
                failed_downloads += 1
        zip_batch(batch_dir)
    print(f"Successfully downloaded {successful_downloads} images.")
    print(f"Failed to download {failed_downloads} images.")

def zip_batch(directory):
    zip_filename = f"{directory}.zip"
    with zipfile.ZipFile(zip_filename, 'w') as zipf:
        for root, dirs, files in os.walk(directory):
            for file in files:
                zipf.write(os.path.join(root, file), file)
    print(f"{zip_filename} created.")

def split_data(df):
    train, test = train_test_split(df, test_size=0.2, random_state=42)
    train, val = train_test_split(train, test_size=0.2, random_state=42)
    return train, val, test

def save_data(df, filename, directory):
    os.makedirs(directory, exist_ok=True)
    df.to_csv(os.path.join(directory, filename), index=False)

if __name__ == "__main__":
    github_url = 'https://raw.githubusercontent.com/Bal67/SkinDetection/main/data/fitzpatrick17k.csv'
    df = load_data_from_github(github_url)
    df = preprocess_data(df)
    download_images(df, batch_size=100)
    train, val, test = split_data(df)
    save_data(train, 'train.csv', '/content/drive/MyDrive/SCIN_Project/data', index=False)
    save_data(val, 'val.csv', '/content/drive/MyDrive/SCIN_Project/data')
    save_data(test, 'test.csv', '/content/drive/MyDrive/SCIN_Project/data')
