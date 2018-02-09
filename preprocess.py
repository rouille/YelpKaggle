import glob
import re
import tqdm
import numpy

from keras.preprocessing import image

def path_to_images(img_dir):
    images = glob.glob(img_dir + '/' + '*.jpg')
    extract_tag = lambda x: int(re.match('.*/([0-9]+).jpg', x).group(1))
    tags = map(extract_tag, images)

    return images, tags

def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size = (224, 224) )
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return numpy.expand_dims(x, axis = 0)
