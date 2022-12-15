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
from tensorflow.keras import optimizers
import argparse

my_parser = argparse.ArgumentParser()
my_parser.add_argument('--fold', type=int, default='', help='training Number of fold(1-5)')

args = my_parser.parse_args()

### '''' Seting '''' <---- ''''
## 📣 Train Fold(1-5)
fold = args.fold
trainfold = f'fold{fold}'
print(f'Trainning Data Set Fold-{fold}')
print(f'-'*100)

## set tensorflow environ
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] ='false'
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR']='platform'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

## set gpu
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="1"
#tf_device='/gpu:1'
#tf_device='/gpu:0'

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

#Setting
BATCH_SIZE = 4
TARGET_SIZE = (480, 480)  # M variant expects images in shape (480, 480)
epochs = 200

# Setting dataset  
dataframe = pd.read_csv('/media/tohn/HDD/VISION_dataset/DATA_OVRDT/CSV/azure_ovrdtDB_Alllevel_6900imgs_round1234_train_split5fold.csv')
dataframe['class'] = dataframe['class'].astype(str)
## train
trainframe = dataframe[dataframe['SplitData'] != trainfold].reset_index(drop=True)
print(f'Train Data Shape [ {trainframe.shape} ]')
#validation
valframe = dataframe[dataframe['SplitData'] == trainfold].reset_index(drop=True)
print(f'Validation Data Shape [ {valframe.shape} ]')
print('-'*100)


#load model
from tensorflow.keras.models import load_model

model_dir = f'/media/tohn/SSD/ModelEfficientV2/OVRDT/All_round/5Class_model/{trainfold}/R1/models/EffnetV2m_R1_5ClassOVRDT_{trainfold}.h5'
model = load_model(model_dir)
height = width = model.input_shape[1]


## Create Data Loader
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
        dataframe = trainframe,
        directory = None,
        x_col = 'path_crop',
        y_col = 'class',
        target_size = (height, width),
        batch_size=BATCH_SIZE,
        color_mode= 'rgb',
        class_mode='categorical')

test_generator = test_datagen.flow_from_dataframe(
        dataframe = valframe,
        directory = None,
        x_col = 'path_crop',
        y_col = 'class',
        target_size = (height, width),
        batch_size=BATCH_SIZE,
        color_mode= 'rgb',
        class_mode='categorical')



## Set TensorBoard 
root_logdir = f'/media/tohn/SSD/ModelEfficientV2/OVRDT/All_round/5Class_model/{trainfold}/R2/Mylogs_tensor/'    ##เปลี่ยน path 
if not os.path.exists(root_logdir) :
    os.makedirs(root_logdir)

def get_run_logdir():
    import time
    run_id = time.strftime("run_%Y_%m_%d_%H_%M_%S")
    return os.path.join(root_logdir,run_id)
run_logdir = get_run_logdir()

tensorboard_cb = callbacks.TensorBoard(log_dir=run_logdir)

#Unfreez
print('This is the number of trainable layers '
          'before freezing the conv base:', len(model.trainable_weights))
model.trainable = True
set_trainable = False
for layer in model.layers:
    if layer.name.startswith('block5'):
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False
print('This is the number of trainable layers '
      'after freezing the conv base:', len(model.trainable_weights))

model.summary()


#Training model    
model.compile(
    optimizer= optimizers.Adam(learning_rate=0.000001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

checkpoint_filepath = f'/media/tohn/SSD/ModelEfficientV2/OVRDT/All_round/5Class_model/{trainfold}/R2/checkpoint/'
if not os.path.exists(checkpoint_filepath) :
        os.makedirs(checkpoint_filepath)
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                                filepath=checkpoint_filepath, save_freq='epoch', ave_weights_only=False)


## Fit model 
model.fit(train_generator,
    epochs=epochs,
    validation_data=test_generator,
    callbacks = [tensorboard_cb, model_checkpoint_callback])

##Save model as TFLiteConverter
modelName = f'EffnetV2m_R2_5ClassOVRDT_{trainfold}'
Pth_model_save = f'/media/tohn/SSD/ModelEfficientV2/OVRDT/All_round/5Class_model/{trainfold}/R2/models/'
if not os.path.exists(Pth_model_save) :
    os.makedirs(Pth_model_save)
    
# Save model as .h5        
model.save(f'{Pth_model_save}{modelName}.h5') 
print(f'Save Model as [ {Pth_model_save}{modelName}.h5 ]')
print('*'*120)

