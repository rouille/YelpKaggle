import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.preprocessing import image


def encode_label(labels):
    target = np.zeros(9, dtype='int')
    for l in labels:
        target[l] = 1
    return target


def decode_label(x):
    return tuple(np.where(x == 1)[0])


def path_to_tensor(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    return np.expand_dims(x, axis=0)


def paths_to_tensor(img_paths):
    tensors = [path_to_tensor(img_path) for img_path in tqdm.tqdm(img_paths)]
    return np.vstack(tensors)


def load_dataset(file):
    data = np.load(file)
    images = data['img']
    targets = data['target']
    return images, targets


def group_images(img, file):
    img2biz = pd.read_csv(file, header = 0, names = ['photo','business'])
    img2biz = img2biz[img2biz['photo'].isin(img)]
    biz2img = pd.DataFrame(img2biz.groupby('business')['photo'].apply(list))

    biz = biz2img.index.values
    img2idx = dict(zip(img,range(len(img))))
    group = []
    for b in biz:
        photos = biz2img.at[b,'photo']
        group.append([img2idx[p] for p in photos])
    return group


def history(model):
    plt.figure(figsize = (15, 5) )

    plt.subplot(121)
    plt.plot(model.history['loss'], color='blue', label='train')
    plt.plot(model.history['val_loss'], color='red', label='valid')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss Function')

    plt.subplot(122)
    plt.plot(model.history['acc'], color = 'blue', label='train')
    plt.plot(model.history['val_acc'], color='red', label='valid')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')


def true_pos(y_true, y_pred):
    return np.sum(y_true * y_pred)


def false_pos(y_true, y_pred):
    return np.sum(y_true * (1. - y_pred))


def false_neg(y_true, y_pred):
    return np.sum((1. - y_true) * y_pred)


def precision(y_true, y_pred):
    return true_pos(y_true, y_pred) / (true_pos(y_true, y_pred) + false_pos(y_true, y_pred))


def recall(y_true, y_pred):
    return true_pos(y_true, y_pred) / (true_pos(y_true, y_pred) + false_neg(y_true, y_pred))


def f1_score(y_true, y_pred):
    return 2. / (1. / recall(y_true, y_pred) + 1. / precision(y_true, y_pred))
