import os
import tensorflow as tf
from efficientnet_v2 import *
import glob
import shutil
import sys
import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt
from tensorflow.keras import callbacks
import pandas as pd
from keras.utils import generic_utils
from efficientnet_v2 import get_preprocessing_layer
from keras import layers
from keras import models

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="0"


# dataset
dataframe = pd.read_csv('/home/yupaporn/codes/USAI/Traindf_fold1_2_viewingAngle.csv')  #Traindf_fold1_2_viewingAngle.csv
dataframe = dataframe[dataframe['Sub_Position_New'] != 'None']
print(f'Train Data Shape [ {dataframe.shape} ]')
#validation
valframe = pd.read_csv('/home/yupaporn/codes/USAI/Validationdf_fold1_2_viewingAngle.csv') #เปลี่ยนตาม fold
valframe = valframe[valframe['Sub_Position_New'] != 'None']
print(f'Validation Data Shape [ {valframe.shape} ]')
DATA_PATH = "/media/tohn/SSD/Images/Image1"
os.chdir(DATA_PATH)
train_dir = os.path.join(DATA_PATH, 'train')
print('-'*100)
print(f'Train Data PATH : [ {train_dir} ]')
#print(train_dir)
validation_dir = os.path.join(DATA_PATH, 'validation')
#print(validation_dir)
print(f'Validation Data PATH : [ {validation_dir} ]')
print('-'*100)

#Setting
BATCH_SIZE = 32
TARGET_SIZE = (480, 480)  # M variant expects images in shape (480, 480)
epochs = 400

## Data Loader
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
      rescale=1./255,
      rotation_range=30,
      width_shift_range=0.2,
      height_shift_range=0.2,
      brightness_range=[0.5,1.5],
      shear_range=0.4,
      zoom_range=0.2,
      horizontal_flip=False,
      fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1./255)
## DataSet
train_generator = train_datagen.flow_from_dataframe(
        dataframe = dataframe,
        directory = train_dir,
        x_col = 'filename',
        y_col = 'Sub_Position_New',
        target_size = TARGET_SIZE,
        batch_size=BATCH_SIZE,
        color_mode= 'rgb',
        class_mode='categorical')

test_generator = test_datagen.flow_from_dataframe(
        dataframe = valframe,
        directory = validation_dir,
        x_col = 'filename',
        y_col = 'Sub_Position_New',
        target_size = TARGET_SIZE,
        batch_size=BATCH_SIZE,
        color_mode= 'rgb',
        class_mode='categorical')

### Train (extract features) Let us fine tune EfficientV2 M variant.
def build_model(num_classes=12):
    base_model = EfficientNetV2M(
        input_shape=(TARGET_SIZE[0], TARGET_SIZE[1], 3),
        include_top=False,
        pooling="avg",
        weights="imagenet-21k"  # Let's use pretrained on imagenet 21k and fine tuned on 1k weight variant.
    )

    base_model.trainable=False  ##freeze the base model. 

    return tf.keras.Sequential([
        base_model,
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(num_classes, activation="softmax")
    ])

model = build_model()
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
    metrics=['accuracy']
)

model.summary()

## Set TensorBoard 
root_logdir = '/media/tohn/SSD/ModelEfficientV2/USAI/ViewingAngle_model/fold1/R1/Mylogs_tensor'

def get_run_logdir():
    import time
    run_id = time.strftime("run_%Y_%m_%d_%H_%M_%S")
    return os.path.join(root_logdir,run_id)
run_logdir = get_run_logdir()

tensorboard_cb = callbacks.TensorBoard(log_dir=run_logdir)

def avoid_error(gen):
    while True:
        try:
            data, labels = next(gen)
            yield data, labels
        except:
            pass

## Fit model 
model.fit(train_generator,
    epochs=epochs,
    validation_data=test_generator,
    callbacks = [tensorboard_cb])

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
# Save
with open("/media/tohn/SSD/ModelEfficientV2/USAI/ViewingAngle_model/fold1/R1/models/V2m_R1_ViewingAngle_fold1_2.tflite", "wb") as file:
      file.write(tflite_model)
#save model        
model.save('/media/tohn/SSD/ModelEfficientV2/USAI/ViewingAngle_model/fold1/R1/models/V2m_R1_ViewingAngle_fold1_2.h5') 
print(f'Save Model as [ /media/tohn/SSD/ModelEfficientV2/USAI/ViewingAngle_model/fold1/R1/models/V2m_R1_ViewingAngle_fold1_2.h5 ]')
