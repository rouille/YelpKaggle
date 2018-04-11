# Import Libraries
import numpy as np
from keras import backend as K
from common import *

# Get Files and Targets
train_ids, train_targets = load_dataset('data/preprocess/train.npz')
valid_ids, valid_targets = load_dataset('data/preprocess/valid.npz')
test_ids, test_targets = load_dataset('data/preprocess/test.npz')

train_files = np.array(['data/train_photos/' + str(i) + '.jpg' for i in train_ids])
valid_files = np.array(['data/train_photos/' + str(i) + '.jpg' for i in valid_ids])
test_files = np.array(['data/train_photos/' + str(i) + '.jpg' for i in test_ids])

nb_train, nb_valid, nb_test = (len(train_ids), len(valid_ids), len(test_ids))
extension = 'all'

# Import the Pre-trained ResNet50 model
from keras.applications.resnet50 import preprocess_input
from keras.applications.resnet50 import ResNet50

model = ResNet50(include_top=False)

# Read Training Dataset
train_features = []
print("Going through training dataset")
for i, file in enumerate(train_files[:nb_train]):
    if i % 10000 == 0: print("%d/%d" % (i, nb_train) )
    tensor = preprocess_input(path_to_tensor(file) )
    train_features.append(model.predict(tensor) )
train_features = np.array(train_features).reshape(len(train_features),1,1,2048)

# Read Validation Dataset
valid_features = []
print("Going through validation dataset")
for i, file in enumerate(valid_files[:nb_valid]):
    if i % 1000 == 0: print("%d/%d" % (i, nb_valid) )
    tensor = preprocess_input(path_to_tensor(file) )
    valid_features.append(model.predict(tensor) )
valid_features = np.array(valid_features).reshape(len(valid_features),1,1,2048)

# Read Test Dataset
test_features = []
print("Going through test dataset")
for i, file in enumerate(test_files[:nb_test]):
    if i % 1000 == 0: print("%d/%d" % (i, nb_test) )
    tensor = preprocess_input(path_to_tensor(file) )
    test_features.append(model.predict(tensor) )
test_features = np.array(test_features).reshape(len(test_features),1,1,2048)

# Write File
print("Saving bottleneck features")
np.savez('data/bottleneck_features/yelp_resnet50_'+extension+'.npz',
         train_features=train_features, train_targets=train_targets[:nb_train],
         valid_features=valid_features, valid_targets=valid_targets[:nb_valid],
         test_features=test_features, test_targets=test_targets[:nb_test])

K.clear_session()
