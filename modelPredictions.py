from keras.preprocessing import image
from keras import models
import numpy as np
import matplotlib.pyplot as plt
import gdown
from pyunpack import Archive
import os

print('Downloading Model...')
link = 'https://drive.google.com/file/d/1ohgLWgpRDoal0qFAJI7xr-Bf7K2NqtUZ/view?usp=sharing'#model
destination = 'model_birds.h5'
gdown.download(url=link, output=destination, quiet=False, fuzzy=True)
print('Finished Download')
model = models.load_model('model_birds.h5')
d = 'finalDataset/'
clList = os.listdir(d)

# Selects 1 random image from a random class, five times
def selectRandomPredictions():
  #print(len(clList))
  rng = np.random.default_rng()
  r1 = rng.choice(70, size=5, replace=False)
  r2 = rng.choice(30, size=5, replace=False)
  imgList = []
  ImgLabel = []
  for i in range(len(r1)):
    #print(clNum, imgNum)
    img = os.listdir(d+clList[r1[i]]+'/')
    imgList.append(d+clList[r1[i]]+'/'+img[r2[i]])
    ImgLabel.append(int(clList[r1[i]][:3]))
  #print(len(imgLst), len(clImg))
  return imgList, ImgLabel

imgList, ImgLabel = selectRandomPredictions()
predLst = []
for i in imgList:
  img = image.load_img(i,  target_size=(150,150))
  img_tensor = image.img_to_array(img)
  img_tensor = np.expand_dims(img_tensor, axis = 0)
  img_tensor /= 255.

  confidence = model.predict(img_tensor)
  predict_class = (confidence > 0.5).astype("int32")
  print (confidence)
  print ("class ", predict_class[0][0], "confindence", )
  predLst.append(predict_class[0][0])
  plt.imshow(img_tensor[0])
  plt.show()