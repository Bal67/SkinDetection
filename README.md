# SkinDetection

Google Colab: https://colab.research.google.com/drive/1rjWW2SNcuJgZAL7w6VqX38Wivb70Zm0U?usp=sharing

FitzPatrick 17k Dataset: https://github.com/mattgroh/fitzpatrick17k

Youtube Link: https://youtu.be/sDVSLhDg57c

AWS S3: https://aws.amazon.com/s3/

This project aims to automatically identify various skin conditions from images, specifically focusing on darker skin tones. The models used are a fine-tuned pre-trained MobileNetV2 model, a non-fine-tuned pre-trained MobileNetV2 model, and an SVM basic model.

## Table of Contents

- [Setup](#setup)

- [Main](#main)

- [scripts](#scripts)

- [models](#models)

- [data](#data)

## Project Structure
setup.py: Script for setting up the environment

app.py: The main Streamlit app

scripts/: Contains the scripts for generating predicting and processing data

dataset.py: Dataset loading and preprocessing

features.py: Processed features from the dataset

non_fine_tuned_model.py: Contains code for non-fine-tuned ml model (non-fine-tuned MobileNetV2)

fine_tuned_model.py: Contains code for fine-tuned ml model (fine-tuned MobileNetV2)

basicmodel.py: Contains code for non-neural network learning model (SVM model)

models/: Contains the saved trained models

data/: Contains the dataset

requirements.txt: List of dependencies

README.md


## Usage
Proceed to the Google Colab page that is linked at the top of this README.md. Once on the page, mount it to your own Google Drive and follow the instructions for each cell in the Google Colab notebook.

Replace all constants in the code (or anywhere where you see a pathway) with the pathway to your local Google Drive folder/Google Drive pathway.

This project uses AWS S3 to store project photos. In the Google Colab, replace ['AWS_SECRET_ACCESS_KEY'] = ___ and ['AWS_ACCESS_KEY_ID'] = ___ with your own AWS Secret Access Key and Access Key ID. Replace ['AWS_REGION'] with the region that your bucket is created in. For this project, ['AWS_REGION'] = 'us-east-1'. Create an AWS S3 bucket or an AWS account if you have not created one. You can follow this link for more information about the S3 bucket: https://aws.amazon.com/s3/

For the Streamlit application: Google Colab has a hard time opening Streamlit applications. To do so, you must run the final cell. At the bottom of that cell will be a link that will lead you to a tunnel website. The bottom cell will also provide you with an IP Address that will look as such (XX.XXX.XXX.XX). Insert that address into the tunnel when prompted for a passcode to access the Streamlit application.

# Model Evaluation

## Evaluation Process and Metric Selection

The evaluation process involves splitting the data into training, validation, and testing sets (70-15-15), training the models, and then evaluating their performance on the test set. The primary metric used for evaluation is Accuracy, precision, and recall, which helped to provide an understanding of the model's ability to correctly classify skin conditions. 

## Data Processing Pipeline

Data Loading: Data is loaded into the script in CSV format.

Feature Extraction: Data is analyzed for relationships. Rows that contained faulty pathways were removed from df. Images were augmented based on the FitzPatrick Scale (<3 = Light; >3 = Dark), with Dark skin tones having more augmentations to the original image.

Data Preparation: Data is split into features and column labels are added. Null values are removed. Data is split into training (70%), validation (15%), and testing sets (15%).

Model Training: The naive, fine-tuned MobileNetV2, and non-fine tuned MobileNetV2 models are trained on the training data, with performance monitored on the validation set.

Model Evaluation: Models are evaluated on the test data and accuracy recorded

# Models Evaluated

SVM Model: Baseline model using SVM 

Naive Model: Non-fine-tuned pre-trained MobileNetV2 model.

  Architecture:
  
    - Embedding Layer
    - Pre-trained MobileNetV2 Backbone
    - Output Layer



Fine-Tuned Model: Fine-tuned MobileNetV2 model.

  Architecture:
    
    - Embedding Layer
    - Pre-trained MobileNetV2 Backbone
    - Fully Connected Layer
    - Dropout Layer
    - Output Layer

  
## Results and Conclusions
SVM Model Accuracy: ~2.3

Naive MobileNetV2 Model Accuracy: ~4.4

Fine-Tuned MobileNetV2 Model Accuracy: ~5.6

The project demonstrates that both naive and fine-tuned NCF models can provide accurate prediction of skin conditions, with the fine-tuned model showing significant improvements in performance. The SVM model serves as a good baseline but is outperformed by the NN models in capturing complex image features.

# Acknowledgments
Data sourced from the GitHub - Matt Groh (https://github.com/mattgroh/fitzpatrick17k)
This project was developed as part of a machine learning course/project.

# Citation
@inproceedings{groh2021evaluating,
  title={Evaluating deep neural networks trained on clinical images in dermatology with the fitzpatrick 17k dataset},
  author={Groh, Matthew and Harris, Caleb and Soenksen, Luis and Lau, Felix and Han, Rachel and Kim, Aerin and Koochek, Arash and Badri, Omar},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={1820--1828},
  year={2021}
}
