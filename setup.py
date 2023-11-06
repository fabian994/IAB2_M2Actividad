import gdown
from pyunpack import Archive
import os
link = 'https://drive.google.com/file/d/1wWgIbkw-aTU6h1i4Zh7E8EF-ySgJQe48/view?usp=sharing'#dataset
destination = 'finalDataset.zip'
gdown.download(url=link, output=destination, quiet=False, fuzzy=True)
Archive(destination).extractall('.')


#Archive(destination).extractall('.')