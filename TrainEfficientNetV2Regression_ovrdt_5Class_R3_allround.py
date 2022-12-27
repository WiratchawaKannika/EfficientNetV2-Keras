import os
import tensorflow as tf
from efficientnet_v2 import *
import glob
import shutil
import sys
import numpy as np
from skimage.io import imread
from tensorflow.keras import callbacks
from keras.callbacks import Callback
import pandas as pd
from keras.utils import generic_utils
from efficientnet_v2 import get_preprocessing_layer
from keras import layers
from keras import models
from tensorflow.keras import optimizers
from keras.optimizers import Adam
import argparse

my_parser = argparse.ArgumentParser()
my_parser.add_argument('--fold', type=int, default='', help='training Number of fold(1-5)')

args = my_parser.parse_args()

### '''' Seting '''' <---- ''''
## 🚗 Train Fold(1-8)
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
BATCH_SIZE = 16
TARGET_SIZE = (480, 480)  # M variant expects images in shape (480, 480)
epochs = 2000
#epochs = 5

# Setting dataset  
dataframe = pd.read_csv('/media/HDD/VISION_dataset/DATA_OVRDT/CSV/azure_ovrdtDB_Alllevel_6900imgs_round1234_train_split5fold.csv')
#dataframe['class'] = dataframe['class'].astype(str)
## train
trainframe = dataframe[dataframe['SplitData'] != trainfold].reset_index(drop=True)
print(f'Train Data Shape [ {trainframe.shape} ]')
#validation
valframe = dataframe[dataframe['SplitData'] == trainfold].reset_index(drop=True)
print(f'Validation Data Shape [ {valframe.shape} ]')
print('-'*100)

#load model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model

model_dir = '/media/SSD/ModelEfficientV2_14p/OVRDT/All_round/5Class_model/fold1/R2/models/EffnetV2m_R2_5ClassOVRDT_fold1_40epoch.h5' #แก้
model = load_model(model_dir)
height = width = model.input_shape[1]

model.summary()


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
      fill_mode='constant')

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
        class_mode='raw')

test_generator = test_datagen.flow_from_dataframe(
        dataframe = valframe,
        directory = None,
        x_col = 'path_crop',
        y_col = 'class',
        target_size = (height, width),
        batch_size=BATCH_SIZE,
        color_mode= 'rgb',
        class_mode='raw')


##Save model 
modelName = f'EffnetV2mRegression_R3_5ClassOVRDT_{trainfold}.h5'

## Set TensorBoard 
root_logdir = f'/media/SSD/ModelEfficientV2_14p/OVRDT_Rrgression/All_round/5Class_model/{trainfold}/R3/Mylogs_tensor/'
#root_logdir = f'/media/SSD/ModelEfficientV2_14p/OVRDT_Regression_test/All_round/5Class_model/{trainfold}/R3/Mylogs_tensor/'
if not os.path.exists(root_logdir) :
    os.makedirs(root_logdir)

def get_run_logdir():
    import time
    run_id = time.strftime("run_%Y_%m_%d_%H_%M_%S")
    return os.path.join(root_logdir,run_id)
run_logdir = get_run_logdir()

tensorboard_cb = callbacks.TensorBoard(log_dir=run_logdir)


def get_preds_and_labels(model, generator):
    """
    Get predictions and labels from the generator
    """
    generator.reset()
    preds = []
    labels = []
    for _ in range(int(np.ceil(generator.samples / BATCH_SIZE))):
        x, y = next(generator)
        preds.append(model.predict(x))
        labels.append(y)
    # Flatten list of numpy arrays
    return np.concatenate(preds).ravel(), np.concatenate(labels).ravel()

class Metrics(Callback):
    def on_epoch_end(self, epochs, logs={}):
        self.model.save(modelName)
        return
    
# For tracking Quadratic Weighted Kappa score and saving best weights
metrics = Metrics()



#Training
model.compile(loss='mse',
              optimizer=Adam(2e-6, decay=1e-6),
              metrics=['mse'])

checkpoint_filepath = f'/media/SSD/ModelEfficientV2_14p/OVRDT_Rrgression/All_round/5Class_model/{trainfold}/R3/checkpoint/'
#checkpoint_filepath = f'/media/SSD/ModelEfficientV2_14p/OVRDT_Regression_test/All_round/5Class_model/{trainfold}/R3/checkpoint/'
if not os.path.exists(checkpoint_filepath) :
        os.makedirs(checkpoint_filepath)
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                                filepath=checkpoint_filepath, save_freq='epoch', ave_weights_only=False)


## Fit model 
model.fit(train_generator,
    epochs=epochs,
    validation_data=test_generator,
    callbacks = [metrics, tensorboard_cb, model_checkpoint_callback])


##Save model 
Pth_model_save = f'/media/SSD/ModelEfficientV2_14p/OVRDT_Rrgression/All_round/5Class_model/{trainfold}/R3/models/'
#Pth_model_save = f'/media/SSD/ModelEfficientV2_14p/OVRDT_Regression_test/All_round/5Class_model/{trainfold}/R3/models/' #เปลี่ยน pathตามfold 
if not os.path.exists(Pth_model_save) :
    os.makedirs(Pth_model_save)

# Save model as .h5        
model.save(f'{Pth_model_save}{modelName}') 
print(f'Save Model as [ {Pth_model_save}{modelName} ]')
print('*'*120)


