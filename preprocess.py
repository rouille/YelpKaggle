import glob
import re
import tqdm
import numpy

from keras.preprocessing import image

def path_to_images(img_dir):
    images = glob.glob(img_dir + '/' + '*.jpg')
    ids = [int(re.match('.*/([0-9]+).jpg', t).group(1)) for t in images]

    return images, ids


def encode_label(labels):
    target = numpy.zeros(9, dtype = 'int')
    for l in labels:
        target[l] = 1
    return target


def decode_label(x):
    return tuple(numpy.where(x == 1)[0])


def path_to_tensor(img_path):
    img = image.load_img(img_path, target_size = (224, 224) )
    x = image.img_to_array(img)
    return numpy.expand_dims(x, axis = 0)


def paths_to_tensor(img_paths):
    tensors = [path_to_tensor(img_path) for img_path in tqdm.tqdm(img_paths)]
    return numpy.vstack(tensors)
