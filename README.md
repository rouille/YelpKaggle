# Yelp Photo Classification

## Overview
[Yelp](https://www.yelp.com/) is a social networking site that publishes crowd-sourced reviews about local businesses. About two years ago, [Kaggle](https://www.kaggle.com/) hosted the Yelp Restaurant Photo Classification challenge. Since labels are optional during the review submission process, some restaurants can be left uncategorized. For this reason, Yelp asked contestants to build an algorithm that automatically predict attributes for restaurants using their user-submitted photographs. The goal of this project is to develop such a model.

## Datasets and Inputs
The photographs and attributes can be found in the data section of the competition [webpage](https://www.kaggle.com/c/yelp-restaurant-photo-classification/data}). Yelp provides for this competition a training dataset and a test dataset. Each image is mapped to a business identification number. The businesses can be tagged with 9 different attributes. The labels are listed below:
1. good_for_lunch
2. good_for_dinner
3. takes_reservations
4. outdoor_seating
5. restaurant_is_expensive
6. has_alcohol
7. has_table_service
8. ambience_is_classy
9. good_for_kids
