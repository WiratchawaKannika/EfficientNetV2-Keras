import os
import tensorflow as tf
from efficientnet_v2 import *
import glob
import shutil
import sys
import numpy as np
from skimage.io import imread
from tensorflow.keras import callbacks
import pandas as pd
from keras.utils import generic_utils
from efficientnet_v2 import get_preprocessing_layer
from keras import layers
from keras import models
from tensorflow.keras import optimizers


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="1"

#Setting
BATCH_SIZE = 32
TARGET_SIZE = (480, 480)  # M variant expects images in shape (480, 480)
epochs = 200

# Setting dataset  
## train
dataframe = pd.read_csv('/media/tohn/HDD/VISION_dataset/DATA_OVRDT/CSV/azure_ovrdtDB_Alllevel_2640imgs_round12_train.csv')
dataframe = dataframe[dataframe['class'] != 5].reset_index(drop=True)
dataframe['class'] = dataframe['class'].astype(str)
print(f'Train Data Shape [ {dataframe.shape} ]')
#validation
valframe = pd.read_csv('/media/tohn/HDD/VISION_dataset/DATA_OVRDT/CSV/azure_ovrdtDB_Alllevel_660imgs_round12_test.csv')
valframe = valframe[valframe['class'] != 5].reset_index(drop=True)
valframe['class'] = valframe['class'].astype(str)
print(f'Validation Data Shape [ {valframe.shape} ]')

## Set Image path 
DATA_PATH = "/media/tohn/HDD/VISION_dataset/DATA_OVRDT/ovrdtDB_08_06_65_from_AjOP_labeled/ovrdtDB"
os.chdir(DATA_PATH)
train_dir = os.path.join(DATA_PATH, 'train')
print('-'*100)
print(f'Train Data PATH : [ {train_dir} ]')
#print(train_dir)
validation_dir = os.path.join(DATA_PATH, 'validation')
#print(validation_dir)
print(f'Validation Data PATH : [ {validation_dir} ]')
print('-'*100)

#load model
from tensorflow.keras.models import load_model

model_dir = '/media/tohn/SSD/ModelEfficientV2/OVRDT/5Class_model/R1_2/models/EffnetV2m_R1_5ClassOVRDT.h5'
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
        dataframe = dataframe,
        directory = train_dir,
        x_col = 'fileName',
        y_col = 'class',
        target_size = (height, width),
        batch_size=BATCH_SIZE,
        color_mode= 'rgb',
        class_mode='categorical')

test_generator = test_datagen.flow_from_dataframe(
        dataframe = valframe,
        directory = validation_dir,
        x_col = 'fileName',
        y_col = 'class',
        target_size = (height, width),
        batch_size=BATCH_SIZE,
        color_mode= 'rgb',
        class_mode='categorical')

## Set TensorBoard 
root_logdir = '/media/tohn/SSD/ModelEfficientV2/OVRDT/5Class_model/R2/Mylogs_tensor/'  ##เปลี่ยน path 
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
    if layer.name.startswith('block7'):
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

checkpoint_filepath = '/media/tohn/SSD/ModelEfficientV2/OVRDT/5Class_model/R2/checkpoint/'
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
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

Pth_model_save = '/media/tohn/SSD/ModelEfficientV2/OVRDT/5Class_model/R2/models/'  ##เปลี่ยน path 
if not os.path.exists(Pth_model_save) :
    os.makedirs(Pth_model_save)
# Save
with open(f"{Pth_model_save}EffnetV2m_R2_5ClassOVRDT.tflite", "wb") as file:
      file.write(tflite_model)
#save model        
model.save(f'{Pth_model_save}EffnetV2m_R2_5ClassOVRDT.h5') 
print(f'Save Model as [ {Pth_model_save}EffnetV2m_R2_5ClassOVRDT.h5 ]')