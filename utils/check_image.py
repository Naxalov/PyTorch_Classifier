import os
import argparse
from pathlib import Path
from PIL import Image
def check_image(fn):
    try:
        im = Image.open(fn)
        exif_data = im._getexif()
        if im.mode!='RGB':
          return False
        im.verify()
        return True
    except:
        return False
    
def check_image_dir(path):
    for fn in path:
        if not check_image(fn):
            print("Corrupt image: {}".format(fn))
            os.remove(fn)

parser = argparse.ArgumentParser(description='DADASET DIR')
parser.add_argument('--dir', default='data/PetImages', type=str, help='dataset folder')
args = parser.parse_args()
DATA_DIR = args.dir
DATA_LIST = list(Path('data/PetImages/').glob('*/*.jpg'))

check_image_dir(DATA_LIST)
#python utils/check_image.py --dir='data/PetImages'