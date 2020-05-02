#!/usr/bin/env python
# coding: utf-8

# In[1]:

import sys
import os
from keras import models
from keras import layers
from keras import optimizers
from keras.applications import VGG16
from keras.applications import InceptionResNetV2
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense, Activation, Input
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras import callbacks, regularizers
from keras.models import load_model
import matplotlib.pyplot as plt 
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras_vggface.vggface import VGGFace
from keras.engine import  Model
from keras.models import load_model
from sklearn import metrics
import numpy


# In[2]:


train_data_path = 'Train'
validation_data_path = 'Validate'
test_data_path = 'Test'
#Parametres
img_width, img_height = 224, 224

# path to the model weights files.
weights_path = 'keras-facenet/weights/facenet_keras_weights.h5'
top_model_weights_path = 'keras-facenet/model/facenet_keras.h5'


epochs = 50
batch_size = 16


# In[3]:


from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint

train_data_path = 'Train'
validation_data_path = 'Validate'
test_data_path = 'Test'
#Parametres
img_width, img_height = 224, 224

#Load the VGG model
vggface = VGGFace(model='resnet50', include_top=False, input_shape=(img_width, img_height, 3))
vggface.summary()
#vgg_model = VGGFace(include_top=False, input_shape=(224, 224, 3))


# In[4]:

# Changing the layer layers
last_layer = vggface.get_layer('avg_pool').output
flattened_layer = Flatten(name='flatten')(last_layer)
densed_layer = Dense(256, activation = 'relu')(flattened_layer)
normalized_layer = BatchNormalization()(densed_layer)
dropout_layer = Dropout(0.5)(normalized_layer)

densed_layer1 = Dense(256, activation = 'relu')(dropout_layer)
normalized_layer1 = BatchNormalization()(densed_layer1)
dropout_layer1 = Dropout(0.5)(normalized_layer1)

densed_layer2 = Dense(256, activation = 'relu')(dropout_layer1)
normalized_layer2 = BatchNormalization()(densed_layer2)
dropout_layer2 = Dropout(0.5)(normalized_layer2)

densed_layer3 = Dense(12, activation='softmax', name='classifier')(dropout_layer2)

custom_vgg_model = Model(vggface.input, densed_layer3)


# Create the model and add the convolutional base model
model = models.Sequential()
model.add(custom_vgg_model)


# In[5]:

# Calculate precision and recall
def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall




# In[6]:


train_datagen = ImageDataGenerator(
      rescale=1./255,
      rotation_range=20,
      width_shift_range=0.2,
      height_shift_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')
 

validation_datagen = ImageDataGenerator(
      rescale=1./255,
      rotation_range=20,
      width_shift_range=0.2,
      height_shift_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')


# In[7]:


train_batchsize = 32
val_batchsize = 32
 
train_generator = train_datagen.flow_from_directory(
        train_data_path,
        target_size=(img_width, img_height),
        batch_size=train_batchsize,
        class_mode='categorical')
 
validation_generator = validation_datagen.flow_from_directory(
        validation_data_path,
        target_size=(img_width, img_height),
        batch_size=val_batchsize,
        class_mode='categorical')


# In[8]:


reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                              patience=100, min_lr=1e-8)

checkpoint = ModelCheckpoint(filepath='checkpoint-{epoch:02d}-{val_loss:.2f}.hdf5', monitor='val_loss', verbose=0, save_best_only=True)


# Compile the model
model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.SGD(lr=1e-4, momentum = 0.9),
              metrics=['acc'])
# Train the model
history = model.fit_generator(
      train_generator,
      steps_per_epoch=train_generator.samples/train_generator.batch_size ,
      epochs=200,
      validation_data=validation_generator,
      validation_steps=validation_generator.samples/validation_generator.batch_size,
      verbose=1, 
      callbacks=[reduce_lr, checkpoint])
 
# Save the model
model.save('keras_vggface_3FC_cropped_300.h5')
