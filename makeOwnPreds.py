import os
import tensorflow as tf
from keras.preprocessing import image
from keras import models
import numpy as np
import matplotlib.pyplot as plt
import gdown
from pyunpack import Archive
import os

print('Downloading Model...')
link = 'https://drive.google.com/file/d/13YsSitwlyERw1Kb_nvn92hwZfV8QilnG/view?usp=sharing'#model
destination = 'xception.h5'
gdown.download(url=link, output=destination, quiet=False, fuzzy=True)
print('Finished Download')
model = models.load_model('xception.h5')
d = 'finalDataset/content/finalDataset/'
clList = ['004.Groove_billed_Ani',
 '007.Parakeet_Auklet',
 '012.Yellow_headed_Blackbird',
 '013.Bobolink',
 '016.Painted_Bunting',
 '017.Cardinal',
 '018.Spotted_Catbird',
 '020.Yellow_breasted_Chat',
 '022.Chuck_will_Widow',
 '024.Red_faced_Cormorant',
 '026.Bronzed_Cowbird',
 '028.Brown_Creeper',
 '029.American_Crow',
 '031.Black_billed_Cuckoo',
 '035.Purple_Finch',
 '036.Northern_Flicker',
 '042.Vermilion_Flycatcher',
 '044.Frigatebird',
 '045.Northern_Fulmar',
 '046.Gadwall',
 '047.American_Goldfinch',
 '049.Boat_tailed_Grackle',
 '053.Western_Grebe',
 '055.Evening_Grosbeak',
 '058.Pigeon_Guillemot',
 '059.California_Gull',
 '068.Ruby_throated_Hummingbird',
 '070.Green_Violetear',
 '071.Long_tailed_Jaeger',
 '073.Blue_Jay',
 '076.Dark_eyed_Junco',
 '077.Tropical_Kingbird',
 '083.White_breasted_Kingfisher',
 '084.Red_legged_Kittiwake',
 '085.Horned_Lark',
 '086.Pacific_Loon',
 '087.Mallard',
 '088.Western_Meadowlark',
 '090.Red_breasted_Merganser',
 '091.Mockingbird',
 '092.Nighthawk',
 '093.Clark_Nutcracker',
 '094.White_breasted_Nuthatch',
 '095.Baltimore_Oriole',
 '099.Ovenbird',
 '101.White_Pelican',
 '102.Western_Wood_Pewee',
 '103.Sayornis',
 '104.American_Pipit',
 '105.Whip_poor_Will',
 '106.Horned_Puffin',
 '108.White_necked_Raven',
 '109.American_Redstart',
 '110.Geococcyx',
 '112.Great_Grey_Shrike',
 '117.Clay_colored_Sparrow',
 '134.Cape_Glossy_Starling',
 '138.Tree_Swallow',
 '140.Summer_Tanager',
 '143.Caspian_Tern',
 '148.Green_tailed_Towhee',
 '150.Sage_Thrasher',
 '151.Black_capped_Vireo',
 '178.Swainson_Warbler',
 '184.Louisiana_Waterthrush',
 '185.Bohemian_Waxwing',
 '191.Red_headed_Woodpecker',
 '193.Bewick_Wren',
 '200.Common_Yellowthroat']


imgT='test.jpg'# write in your image path
img = tf.keras.utils.load_img(
    imgT, target_size=(224,224))
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)
img_array /= 255.
predictions = model.predict(img_array)
print('max value ',predictions[0].max())
print('max val pos ',predictions[0].argmax())
print("Predicted: ", clList[predictions[0].argmax()])
plt.imshow(img_array[0])# correct pred is cardinal
plt.show()