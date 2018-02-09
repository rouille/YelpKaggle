import glob
import re

def get_files(path):
    files = glob.glob(path + '/' + '*.jpg')
    extract_tag = lambda x: int(re.match('.*/([0-9]+).jpg', x).group(1))
    tags = map(extract_tag, files)

    return files, tags
