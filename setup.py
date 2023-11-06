import gdown
from pyunpack import Archive
import os
link = 'https://drive.google.com/file/d/1L389q1jTTWNZ0dqvYXwyu8TuE6Gmg3Hq/view?usp=sharing'#dataset
destination = 'finalDataset.zip'
gdown.download(url=link, output=destination, quiet=False, fuzzy=True)
Archive(destination).extractall('.')


#Archive(destination).extractall('.')