# A01367585 Fabian Gonzalez Vera
# CNN for multiclass clasification using Transfer learning from model VGG16

import tensorflow
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing import image
import pandas as pd

def selectRandomPredictions():
  # Function selects 5 random images from 5 random classes, 1 images per class
  d = 'finalDataset/'
  clList = os.listdir(d)
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

# Previously extracted labels
labels = [ 4,   7,  12,  13,  16,  17,  18,  20,  22,  24,  26,  28,
        29,  31,  35,  36,  42,  44,  45,  46,  47,  49,  53,  55,  58,
        59,  68,  70,  71,  73,  76,  77,  83,  84,  85,  86,  87,  88,
        90,  91,  92,  93,  94,  95,  99, 101, 102, 103, 104, 105, 106,
       108, 109, 110, 112, 117, 134, 138, 140, 143, 148, 150, 151, 178,
       184, 185, 191, 193, 200]

img_height,img_width = 150, 150
batch_size = 20

#Data augmentation & generators
datagen = ImageDataGenerator( 
    rescale = 1./255,
    rotation_range = 40,
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    horizontal_flip = True,
    vertical_flip = True,
    validation_split=0.2,
    )

train_generator = datagen.flow_from_directory(
    'finalDataset/',
    target_size = (img_height,img_width),
    batch_size = batch_size,
    class_mode ='sparse',
    shuffle=True,
    subset='training'
    )

train_generator = datagen.flow_from_directory(
    'finalDataset/',
    target_size = (img_height,img_width),
    batch_size = batch_size,
    class_mode ='sparse',
    shuffle=True,
    subset='training', seed = 123
    )

# validation generator
val_datagen = ImageDataGenerator(1./255)

val_generator = val_datagen.flow_from_directory(
							'finalDataset/',
							target_size = (150,150),
							batch_size =20,
							class_mode= 'sparse')

# test generator
test_datagen = ImageDataGenerator(1./255)

test_generator = test_datagen.flow_from_directory('finalDataset',
                                                 shuffle=False,
                                                 batch_size=20,
                                                 target_size = (150,150),
                                                 class_mode='sparse')


# Model
model = models.Sequential()

conv_base = VGG16(weights='imagenet',
								include_top = False,
								input_shape = (150,150,3))

model.add(conv_base)
conv_base.trainable = False
#model.add(layers.Conv2D(16, kernel_size=3, padding='same', strides = 3, activation="relu", input_shape=(150,150,3)))
#model.add(layers.MaxPooling2D((3,3)))
#model.add(layers.Conv2D(16, kernel_size=4, padding='same', strides = 2, activation="relu", input_shape=(150,150,3)))
model.add(layers.MaxPooling2D((3,3)))
model.add(layers.Flatten())
model.add(layers.Dense(256,activation='relu'))
model.add(layers.Dense(1,activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
						optimizer=optimizers.Adam(learning_rate=.005),
						metrics=['acc'])

history = model.fit(train_generator,
				steps_per_epoch = 75,
				epochs = 10,
				validation_data = val_generator,
				validation_steps = 25)

model.save('model_birds.h5')



acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc)+1)

plt.plot(epochs,acc,'bo',label='train accuracy')
plt.plot(epochs,val_acc, 'b', label='validation accuracy')
plt.title('train acc vs val acc')
plt.legend()

plt.figure()

plt.plot(epochs,loss, 'bo', label ='training loss')
plt.plot(epochs,val_loss, 'b', label = 'validation loss')
plt.title('train loss vs val loss')
plt.legend()

plt.show()


# Test some predictions

print('Some predictions: ')
predictions = model.predict(test_generator)

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


print('Confusion Matrix: ')
y_pred = np.argmax(predictions, axis=1)
y_true = test_generator.classes

cm = confusion_matrix(y_true, y_pred,labels=labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot()
plt.show()