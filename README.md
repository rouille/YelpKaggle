# Yelp Photo Classification

## Overview
About two years ago, [Kaggle](https://www.kaggle.com/) hosted the Yelp Restaurant Photo Classification challenge. For those who are unaware, [Yelp](https://www.yelp.com/) is a social networking site that publishes crowd-sourced reviews about local businesses. Since labels are optional during the review submission process and, for this reason, some restaurants can be left uncategorized.

Yelp asked contestants to build an algorithm that automatically predict attributes for restaurants using their user-submitted photographs. The goal of this project is to develop such a model.


## Datasets and Inputs
All the photographs and attributes can be found in the data section of the competition on Kaggle (click [here](https://www.kaggle.com/c/yelp-restaurant-photo-classification/data)). Each image is mapped to a business identification number and each business is tagged as follow:
1. good_for_lunch
2. good_for_dinner
3. takes_reservations
4. outdoor_seating
5. restaurant_is_expensive
6. has_alcohol
7. has_table_service
8. ambience_is_classy
9. good_for_kids

This is a multi-instance multi-label (MIML) classification problem. Each business has multiple photographs (117 images on average, see [eda.ipynb](eda.ipynb)) and predictions need to be made at the business level. We have then two options.

The first one is to derive a feature vector for each instance and combine them accordingly to get one feature vector per business. These inputs can then be used in a standard supervised learning task. The second option is to assign to each instance the label of its corresponding business and proceed to classification. Then, the output probabilities are averaged for each label. We have investigated both options.

Also, multiple labels can be assigned to each business. This means it is necessary to take the label dependencies into account for classification.


## Files Description
You will find below a short description of what we tackle in the various files available in this repository.
* [`eda.ipynb`](eda.ipynb): exploratory data analysis. This where we get familiar with the datasets. Data are loaded, statistics are derived, photographs are displayed, etc.
* [`split_data.ipynb`](split_data.ipynb): preprocessing. The training dataset (234,842 photographs) is split into a training (75%), validation (12.5%) and test (12.5%) datasets.
* [`bottleneckFeaturesExtraction.ipynb`](bottleneckFeaturesExtraction.ipynb): bottleneck features from state-of-the-art pre-trained deep learning models are extracted. For a small fraction of the available training (20,000 tensors), validation (2,000 tensors) and test data (2,000 tensors), we store the last activation map before the fully connected layers for the *VGG16*, *Xception*, *ResNet50* and *InceptionV3* models.
* [`pretrainedModelsComparison.ipynb`](pretrainedModelsComparison.ipynb): model comparison. The bottleneck features of the various models extracted previously are taken as input of a very simple neural network. The idea here is to compare the performance of the pre-trained models using a simple F1 score at the instance level. *ResNet50* is the best performing model and will be used as a bottleneck feature extractor in this project.
* [`bottleneckFeaturesExtraction_resnet50.py`](bottleneckFeaturesExtraction_resnet50.py): Get bottleneck features for the pre-trained *ResNet50* deep learning model.
* [`trees.ipynb`](xgboost.ipynb): boosted trees are used for classification. The *XGBoost* software library is used for this purpose. The *ResNet50* bottleneck features are grouped by business and averaged to obtain a single feature vector for each business. Each of these inputs are then reduced to 200 features using a principal component analysis. Two analyses are then carried out for classifying images. The first scenario is to ignore the label dependencies. In this case, boosted trees are trained on each class independently and the F1 score is calculated on the test dataset: **0.76503**. The second scenario consists in exploiting the dependencies using a [classifier chain](https://en.wikipedia.org/wiki/Classifier_chains). Using the test dataset, the F1 score is then: **0.78695**.
* [`nn.ipynb`](nn.ipynb): multi-output neural network is used for classification. The model is trained on the *ResNet50* bottleneck features. Predictions are then grouped by business and averaged to obtain a single prediction for for each business. The F1 score is calculated on the test dataset: **0.83160**.
* [`findBestThreshold.ipynb`](findBestThreshold.ipynb): determine the optimal threshold for converting the probabilities obtained after classification -- as returned by the sigmoid activation function of the NN -- to labels (0 or 1). Matthews correlation coefficients are used to find the best threshold for each label.
* [`submission.ipynb`](submission.ipynb): Generate files for submission. The custom.csv file has been submitted. F1 score is **0.81150** based on the private dataset. Ranked 63<sup>rd</sup> over 355.
* [`common.py`](common.py): Functions used across the various notebooks.


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


## Running the Notebooks
Bottleneck features and neural networks have been respectively calculated and trained with a GPU compute instance on AWS (Amazon Web Services). The Deep Learning AMI available in the AWS Marketplace has been used. Once logged on the EC2 instance, The TensorFlow (Python 3) environment is selected. Do not forget to add a custom TCP rule to be able to reach the notebook. Usually, Jupyter notebook runs on port 8888. 
