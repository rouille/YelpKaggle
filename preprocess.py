import glob
import re

def path_to_images(img_dir):
    images = glob.glob(img_dir + '/' + '*.jpg')
    extract_tag = lambda x: int(re.match('.*/([0-9]+).jpg', x).group(1))
    tags = map(extract_tag, images)

    return images, tags
