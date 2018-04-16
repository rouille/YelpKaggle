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

Note that this is a multi-instance multi-label (MIML) classification problem. Each business has multiple photographs (117 images on average, see [eda.ipynb](eda.ipynb)) and predictions ned to be made at the business level. One option is to first obtain a features vector for each instance and then combine them accordingly. In this case, we will have one feature vector for each of the 2,000 and 10,000 businesses in the training and test datasets, respectively, which will be used in a standard supervised learning task. The other option is to map each instance to the label of its corresponding business and proceed to classification. Then, for each label, the output probabilities can be averaged. Multiple labels can be assigned to each business. This means it will be necessary to take the label dependencies into account for classification.

## What's in the Repository and Summary of Results
Here is a short description of the different files available in this directory:
* [`eda.ipynb`](eda.ipynb): exploratory data analysis. This where we get familiar with the datasets. Data are loaded, statistics are derived, photographs are displayed, etc.
* [`split_data.ipynb`](split_data.ipynb): preprocessing. The training dataset (234,842 photographs) is split into a training (75%), validation (12.5%) and test (12.5%) datasets.
* [`bottleneckFeaturesExtraction.ipynb`](bottleneckFeaturesExtraction.ipynb): bottleneck features from state-of-the-art pre-trained deep learning models are extracted. For a small fraction of the available training (20,000 tensors), validation (2,000 tensors) and test data (2,000 tensors), we store the last activation map before the fully connected layers for the *VGG16*, *Xception*, *ResNet50* and *InceptionV3* deep learning models.
* [`pretrainedModelsComparison.ipynb`](pretrainedModelsComparison.ipynb): model comparison. The bottleneck features calculated in [bottleneckFeaturesExtraction.ipynb](bottleneckFeaturesExtraction.ipynb) are taken as input of a very simple Convolutional Neural Network (CNN). The idea here is to compare the performance of the four pre-trained models using a simple F1 score. *VGG16*: **0.75284**, *Xception*: **0.74756**, *ResNet50*: **0.76744**, *InceptionV3*: **0.74086**
* [`xgboost.ipynb`](xgboost.ipynb): classification with the *XGBoost* algorithm. Each of the bottleneck features calculated with ResNet50 in [bottleneckFeaturesExtraction.ipynb](bottleneckFeaturesExtraction.ipynb) are reduced to 200 features using a principal component analysis. Two analyses are then carried out for classifying images. The first scenario is to ignore the label dependencies. In this case, the *XGBoost* model is trained on each class independently and the F1 score is derived: **0.75027**. The second scenario consists in exploiting the dependencies using a [classifier chain](https://en.wikipedia.org/wiki/Classifier_chains). The F1 score is then calculated: **0.76238**.
* [`finetune_resnet50.ipynb`](finetune_resnet50.ipynb): fine-tune *ResNet50*. The last blocks of *ResNet50* along with a simple neural network plugged on top are trained. A F1 score of **0.78925** is reached. Here, 50,000 training tensors are considered. Both, the validation and test datasets are composed of 5,000 photographs.
* [`f1score.ipynb`](f1score.ipynb): determine the optimal threshold for converting the probabilities obtained after classification -- as returned by the sigmoid activation function of the NN -- to labels (0 or 1). The best threshold is in the range [0.40,0.45].
* [`bottleneckFeaturesExtraction_resnet50.py`](bottleneckFeaturesExtraction_resnet50.py): Get bottleneck features for the pre-trained *ResNet50* deep learning model.
* [`common.py`](common.py): Functions used in the various notebooks.

## Software and Libraries
Python 3.5 is used for this project. Various libraries are used along the project. These include:
* sklearn 0.19.1
* keras 2.0.0. TensorFlow is used as a backend.
* xgboost 0.7.post3
* pandas 0.19.2
* seaborn 0.7.1
* PIL 4.0.0
* numpy 1.12.1
* matplotlib 2.0.0
* tqdm 4.11.2
