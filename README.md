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

## What's in this repository
Here is a short description of the different files available in this directory:
* `eda.ipynb`: exploratory data analysis. This where we get familiar with the datasets. Data are loaded, statistics are derived, photographs are displayed, etc.
* `split_data.ipynb`: preprocessing. The training dataset (234842 photographs) is split into a training (75%), validation(12.5%) and test (12.5%) datasets.
* `bottleneck.ipynb`: bottleneck features from state-of-the-art pre-trained deep learning models. For a small fraction of the available training, validation and test data, we store the last activation map before the fully connected layers for the VGG16, Xception and ResNet50 and InceptionV3 deep learning models.
* `compare.ipynb`: model comparison. The bottleneck features  calculated in the previous notebook are taken as input of a very simple Convolutional Neural Network (CNN). The idea here is to compare the performance of the four pre-trained models using a simple F1 score.
* `common.py`: Functions used in the various notebooks.
