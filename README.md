# Yelp Photo Classification

## Overview
About two years ago, [Kaggle](https://www.kaggle.com/) hosted the Yelp Restaurant Photo Classification challenge. For those who are unaware, [Yelp](https://www.yelp.com/) is a social networking site that publishes crowd-sourced reviews about local businesses. Since labels are optional during the review submission process and, for this reason, some restaurants can be left uncategorized.

Yelp asked contestants to build an algorithm that automatically predict attributes for restaurants using their user-submitted photographs. The goal of this project is to develop such a model.

## Datasets and Inputs
All the photographs and attributes can be found in the data section of the competition on Kaggle (click [here](https://www.kaggle.com/c/yelp-restaurant-photo-classification/data)).

Each image is mapped to a business identification number. The businesses can be tagged with 9 different labels:
1. good_for_lunch
2. good_for_dinner
3. takes_reservations
4. outdoor_seating
5. restaurant_is_expensive
6. has_alcohol
7. has_table_service
8. ambience_is_classy
9. good_for_kids

## What's in the Repository and Summary of Results
Here is a short description of the different files available in this directory:
* `eda.ipynb`: exploratory data analysis. This where we get familiar with the datasets. Data are loaded, statistics are derived, photographs are displayed, etc.
* `split_data.ipynb`: preprocessing. The training dataset (234,842 photographs) is split into a training (75%), validation(12.5%) and test (12.5%) datasets.
* `bottleneck.ipynb`: bottleneck features from state-of-the-art pre-trained deep learning models. For a small fraction of the available training (20,000 tensors), validation (2,000 tensors) and test data (2,000 tensors), we store the last activation map before the fully connected layers for the *VGG16*, *Xception*, *ResNet50* and *InceptionV3* deep learning models.
* `compare.ipynb`: model comparison. The bottleneck features calculated in the previous notebook are taken as input of a very simple Convolutional Neural Network (CNN). The idea here is to compare the performance of the four pre-trained models using a simple F1 score. *VGG16*: **0.75284**, *Xception*: **0.74756**, *ResNet50*: **0.76744**, *InceptionV3*: **0.74086**
* `xgboost.ipynb`: classification with the *XGBoost* algorithm. Each of the bottleneck features calculated with ResNet50 in `bottleneck.ipynb` are reduced to 200 features using a principal component analysis. Two analyses are then carried out for the classification. The independence case where the correlation among classes are ignored. The *XGBoost* model is then trained on each class independently. The F1 score is **0.75027**. The correlations can be exploited using a [classifier chain](https://en.wikipedia.org/wiki/Classifier_chains). The F1 score is **0.76238**.
* `finetuning.ipynb`: fine-tune *ResNet50*. The last blocks of *ResNet50* along with a simple CNN plugged on top of it are trained. A F1 score of **0.78466** is reached. Here, 50,000 training tensors are considered. Both, the validation and test datasets are composed of 5,000 photographs.
* `common.py`: Functions used in the various notebooks.
