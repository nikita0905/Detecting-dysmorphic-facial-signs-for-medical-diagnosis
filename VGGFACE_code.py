#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras import models
from keras import layers
from keras import optimizers
from keras.applications import VGG16
from keras.applications import InceptionResNetV2
import sys
import os
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

#nb_train_samples = 1774
#nb_validation_samples = 313
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
#vgg_conv = VGG16(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))

vggface = VGGFace(model='resnet50', include_top=False, input_shape=(img_width, img_height, 3))
vggface.summary()
#vgg_model = VGGFace(include_top=False, input_shape=(224, 224, 3))


# In[4]:


last_layer = vggface.get_layer('avg_pool').output
x = Flatten(name='flatten')(last_layer)
xx = Dense(256, activation = 'relu')(x)
x1 = BatchNormalization()(xx)
x2 = Dropout(0.5)(x1)

y = Dense(256, activation = 'relu')(x2)
yy = BatchNormalization()(y)
y1 = Dropout(0.5)(yy)

z = Dense(256, activation = 'relu')(y1)
zz = BatchNormalization()(z)
z1 = Dropout(0.5)(zz)

x3 = Dense(12, activation='softmax', name='classifier')(z1)

custom_vgg_model = Model(vggface.input, x3)


# Create the model
model = models.Sequential()
 
# Add the convolutional base model
model.add(custom_vgg_model)


# In[5]:


def precision(y_true, y_pred):
    """Precision metric.
    Only computes a batch-wise average of precision.
    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall(y_true, y_pred):
    """Recall metric.
    Only computes a batch-wise average of recall.
    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


# In[6]:


def fbeta_score(y_true, y_pred, beta=1):
    """Computes the F score.
    The F score is the weighted harmonic mean of precision and recall.
    Here it is only computed as a batch-wise average, not globally.
    This is useful for multi-label classification, where input samples can be
    classified as sets of labels. By only using accuracy (precision) a model
    would achieve a perfect score by simply assigning every class to every
    input. In order to avoid this, a metric should penalize incorrect class
    assignments as well (recall). The F-beta score (ranged from 0.0 to 1.0)
    computes this, as a weighted mean of the proportion of correct class
    assignments vs. the proportion of incorrect class assignments.
    With beta = 1, this is equivalent to a F-measure. With beta < 1, assigning
    correct classes becomes more important, and with beta > 1 the metric is
    instead weighted towards penalizing incorrect class assignments.
    """
    if beta < 0:
        raise ValueError('The lowest choosable beta is zero (only precision).')

    # If there are no true positives, fix the F score at 0 like sklearn.
    if K.sum(K.round(K.clip(y_true, 0, 1))) == 0:
        return 0

    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    bb = beta ** 2
    fbeta_score = (1 + bb) * (p * r) / (bb * p + r + K.epsilon())
    return fbeta_score


def fmeasure(y_true, y_pred):
    """Computes the f-measure, the harmonic mean of precision and recall.
    Here it is only computed as a batch-wise average, not globally.
    """
    return fbeta_score(y_true, y_pred, beta=1)


# In[7]:


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


# In[8]:


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


# In[9]:


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


# In[ ]:




