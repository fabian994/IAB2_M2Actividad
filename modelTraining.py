# A01367585 Fabian Gonzalez Vera
# CNN for multiclass clasification using a Bilinear CNN based in the paper Bilinear CNN Models for Fine-grained Visual Recognition by Tsung-Yu Lin et al.

import tensorflow
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_datasets as tfds
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
import seaborn as sns
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.applications import VGG16
import IPython.display as display
from PIL import Image
import cv2
from sklearn.metrics import accuracy_score, precision_score, recall_score

def selectRandomPredictions():
  # Function selects 5 random images from 5 random classes, 1 images per class
  d = 'finalDataset/content/finalDataset/'
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

# Create dataset
directory = 'finalDataset/content/finalDataset/'
trainds, valds = tf.keras.utils.image_dataset_from_directory(directory,                  
    batch_size=25,
    image_size=(224, 224),
    shuffle=True,
    seed=123,
    validation_split=0.2,
    subset='both')
class_names = trainds.class_names

# Change image size
size = (224, 224)
IMG_SIZE = 224
trainds = trainds.map(lambda x, y: (tf.image.resize(x, size), y))
valds = valds.map(lambda x, y: (tf.image.resize(x, size), y))


data_augmentation = keras.Sequential(
    [ 
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.3),
    ]
)
units =len(class_names)

# Dnet
DNet = keras.applications.Xception(weights='imagenet',
								include_top = False,
								input_shape = (224,224,3))
DNet.trainable = False
DNet._name = 'DNet'

# M net

MNet = keras.applications.Xception(weights='imagenet',
								include_top = False,
								input_shape = (224,224,3))
MNet.trainable = False
MNet._name = 'MNet'

# outer product
def outer_product(x):
    #outer product
    op = tf.einsum('ijkm,ijkn->imn',x[0],x[1])
    
    # Reshape from [batch_size,depth,depth] to [batch_size, depth*depth]
    op = tf.reshape(op,[-1,x[0].shape[3]*x[1].shape[3]])
    
    # Divide by feature map size [sizexsize]
    size1 = int(x[1].shape[1])
    size2 = int(x[1].shape[2])
    op = tf.divide(op, size1*size2)
    
    # Take signed square root of phi_I
    y_ssqrt = tf.multiply(tf.sign(op),tf.sqrt(tf.abs(op)+1e-12))
    
    # Apply l2 normalization
    z_l2 = tf.nn.l2_normalize(y_ssqrt, axis=1)
    return z_l2

# model
inputs = keras.Input(shape=(224, 224, 3),dtype='float32')
BCNN = data_augmentation(inputs)
scale_layer = keras.layers.Rescaling(scale=1 / 155, offset=-1)
x = scale_layer(BCNN)

#x = MNet(x, training=False)

CNNA = DNet(x, training=False)
CNNB = MNet(x, training=False)

#CNNA = CNNA[:,:-1,:-1,:]
bilinear = keras.layers.Lambda(outer_product, name='outer_product1')([CNNA,CNNB])


# Flatten
op = keras.layers.Flatten(name='flatten')(bilinear)#fl

# Softmax
outputs = keras.layers.Dense(units,name='softmax',activation='softmax')(op)

model2 = keras.Model(inputs, outputs)

model2.summary()

model2.compile(loss='sparse_categorical_crossentropy',
						optimizer=keras.optimizers.Adam(learning_rate=.03),
						metrics=['acc'])

history = model2.fit(trainds,
				#steps_per_epoch = 75,
				epochs = 70,
				validation_data = valds,
				validation_steps = 10)

model2.save('xception.h5')

frame = pd.DataFrame(history.history)

acc = frame['acc']
val_acc = frame['val_acc']
loss = frame['loss']
val_loss = frame['val_loss']

epochs = range(1, len(acc)+1)

#acc_plot = frame.plot(y="val_loss", title = "Loss vs Epochs",legend=False)
#acc_plot.set(xlabel="Epochs", ylabel="Loss")

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

acc_plot = frame.plot(y="acc", title="Accuracy vs Epochs", legend=False)
acc_plot.set(xlabel="Epochs", ylabel="Accuracy")

# metrics

y_pred = []  # store predicted labels
y_true = []  # store true labels

# iterate over the dataset
for image_batch, label_batch in valds:   # use dataset.unbatch() with repeat
   # append true labels
   y_true.append(label_batch)
   # compute predictions
   preds = model2.predict(image_batch)
   # append predicted labels
   y_pred.append(np.argmax(preds, axis = - 1))

# convert the true and predicted labels into tensors
correct_labels = tf.concat([item for item in y_true], axis = 0)
predicted_labels = tf.concat([item for item in y_pred], axis = 0)

yTrue = []
YPred = []
for i in correct_labels:
    yTrue.append(class_names[i])
for i in predicted_labels:
    YPred.append(class_names[i])

print("Overall Accuracy:",accuracy_score(yTrue, YPred))
print("Overall Precision:",precision_score(yTrue, YPred, average='macro'))
print("Overall Recall:",recall_score(yTrue, YPred, average='macro'))


cm = confusion_matrix(yTrue, YPred)
fig = plt.figure(figsize = (50,50))
ax1 = fig.add_subplot(1,1,1)
sns.set(font_scale=1.2) #for label size
sns.heatmap(cm, annot=True, annot_kws={"size": 12},
     cbar = False, cmap='Purples')#,xticklabels=class_names, yticklabels=class_names);
ax1.set_ylabel('True Values',fontsize=12)
ax1.set_xlabel('Predicted Values',fontsize=12)
plt.savefig('confM_xcep.png', dpi = 300)
plt.show()
