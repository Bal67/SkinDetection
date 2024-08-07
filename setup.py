# VideoGameRec

Google Colab: https://colab.research.google.com/drive/1l9dR4gdm-N6baCLi6J4YWkPsyECgfi9c?usp=sharing

Kaggle Dataset: https://www.kaggle.com/datasets/tamber/steam-video-games/data

Youtube Link: https://youtu.be/6fKk6E7ADhg

This project ___. The models used are a fine-tuned Neural Collaborative Filtering (NCF) model, a non-fine-tuned NCF model, and a KNN basic model.

## Table of Contents

- [Setup](#setup)

- [Main](#main)

- [scripts](#scripts)

- [models](#models)

- [data](#data)

## Project Structure
setup.py: Script for setting up the environment

app.py: The main Streamlit app

scripts/: Contains the scripts for generating recommendations and processing data

dataset.py: Dataset loading and preprocessing

features.py: Processed features from the dataset

non_fine_tuned_model.py: Contains code for non fine-tuned ml model (non fine-tuned MobileNetV2)

fine_tuned_model.py: Contains code for fine-tuned ml model (fine-tuned MobileNetV2)

basicmodel.py: Contains code for non-neural network learning model (SVM model)

models/: Contains the saved trained models

data/: Contains the dataset

requirements.txt: List of dependencies

README.md


## Usage
Proceed to the Google Colab page that is linked at the top of this README.md. Once on the page, mount it to your own Google Drive and follow the instructions for each cell in the Google Colab notebook.

Replace all constants in the code (or anywhere where you see a pathway) with the pathway to your local Google Drive folder/Google Drive pathway.

For the Streamlit application: Google Colab has a hard time opening Streamlit applications. To do so, you must run the final cell. At the bottom of that cell will be a link that will lead you to a tunnel website. The bottom cell will also provide you with an IP Address that will look as such (XX.XXX.XXX.XX). Insert that address into the tunnel when prompted for a passcode to access the Streamlit application.

# Model Evaluation

## Evaluation Process and Metric Selection

The evaluation process involves splitting the data into training, validation, and testing sets (60-20-20), training the models, and then evaluating their performance on the test set. The primary metric used for evaluation is RMSE (Root Mean Square Error), which measures the differences between predicted and actual ratings. 

## Data Processing Pipeline

Data Loading: Data is loaded into the script in CSV format.

Feature Extraction: Data is analyzed for relationships. Additional columns are created for "Average hour played" for player and game, "Game Rating", and more

Data Preparation: Data is split into features and column labels are added. Null values are removed. Data is split into training (60%), validation (20%), and testing sets (20%).

Model Training: The naive, fine-tuned NCF, and KNN models are trained on the training data, with performance monitored on the validation set.

Model Evaluation: Models are evaluated on the test data and RMSE recorded

# Models Evaluated

KNN Model: Baseline model using K-Nearest Neighbors for collaborative filtering.

Naive Model: Non-fine-tuned Neural Collaborative Filtering (NCF) model.

  Architecture:
  
    - Embedding Layer
    - Flatten Layer
    - Dot Product Layer
    - Dense Layer


Fine-Tuned NCF Model: Fine-tuned Neural Collaborative Filtering model with hyperparameter tuning and additional layers.

  Architecture:
    
    - Embedding Layer
    - Flatten Layer
    - Concatenate Layer
    - Dense Layers with Dropout and L2 regularization

  
## Results and Conclusions
KNN Model RMSE: ~3.3

Naive Neural Collaborative Filtering Model RMSE: ~1.8

Fine-Tuned Neural Collaborative Filtering Model RMSE: ~1.6

The project demonstrates that both naive and fine-tuned NCF models can provide accurate game recommendations, with the fine-tuned model showing significant improvements in performance. The KNN model serves as a good baseline but is outperformed by the NCF models in capturing complex user-game interactions.

# Acknowledgments
Data sourced from the Steam video games dataset on Kaggle. 
This project was developed as part of a machine learning course/project.

