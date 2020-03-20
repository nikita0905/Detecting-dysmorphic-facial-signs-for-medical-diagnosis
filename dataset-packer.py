#!/usr/bin/env python3

'''

'''

import os
import sys
import traceback
import numpy as np
import matplotlib.image as mpimg

from os import listdir
from os.path import isfile, join, isdir


def extract(srcDir,images,label,recCount):
    """
    Recursively read in all images and generate labels
    images is in format of { key: { 'img': np.array, 'dir': string, 'lbl': string }}
    labels are generated from the directory hierarchy, appended by '_'.
    """

    if recCount < 0:
        return images

    for f in listdir(srcDir):
        if isdir(join(srcDir,f)):
            images = extract(join(srcDir,f),images,label+'_'+f,recCount-1)
        elif isfile(join(srcDir,f)) and f != 'desktop.ini':
            images[f] = {}
            images[f]['lbl'] = label
            images[f]['dir'] = srcDir
            origImg = mpimg.imread(join(srcDir,f))
            images[f]['img'] = origImg

    return images


def preprocessing(images):
    #normX, normY = get_max_dims(images)
    #aveX, aveY = get_ave_dims(images)
    #print(normX,normY)
    #print(aveX,aveY)
    #return normalize_images(images,normX,normY)
    return images

def get_max_dims(images):
    maxX = 0
    maxY = 0

    for imgName in images.keys():
        img = images[imgName]['img']

        if maxX < img.shape[1]:
            maxX = img.shape[1]

        if maxY < img.shape[0]:
            maxY = img.shape[0]

    return maxX, maxY

def get_ave_dims(images):
    sumX = 0
    sumY = 0
    count = len(images.keys())

    for imgName in images.keys():
        img = images[imgName]['img']

        sumX = sumX + img.shape[1]
        sumY = sumY + img.shape[0]

    return sumX/count, sumY/count

def enforce_RGBA_channels(img):
    """
    Convert an image to RGBA if possible.
    Cases supported:
        + Already in RGBA
        + In RGB
        + In Grayscale (single-channel)
    """

    if len(img.shape) < 3 or img.shape[2] < 4:
        if len(img.shape) == 2 or img.shape[2] == 1:
            # gray-scale to average RGB + full A
            alpha = np.ones((img.shape[0],img.shape[1],1))
            if len(img.shape) == 2:
                img = np.reshape(img,(img.shape[0],img.shape[1],1))
            img = np.concatenate((img,img,img,alpha), axis=2)
        elif len(img.shape) < 2 or img.shape[2] != 3:
            raise Exception('Image is not standard size')
        else:
            # RGB image
            alpha = np.ones((img.shape[0],img.shape[1],1))
            img = np.concatenate((img,alpha),axis=2)
    elif img.shape[2] > 4:
        print(img.shape)
        raise Exception('Image is not standard size')
    return img


def crop_image_x(img, normX):
    """
    Crops image along X direction (balanced)
    """

    diff = img.shape[1] - normX
    startX = diff//2
    if diff % 2 != 0:
        startX += 1
    stopX = img.shape[1] - diff//2

    return img[:,startX:stopX]


def crop_image_y(img, normY):
    """
    Crops image along Y direction (balanced)
    """

    diff = img.shape[0] - normY
    startY = diff//2
    if diff % 2 != 0:
        startY += 1
    stopY = img.shape[0] - diff//2

    return img[startY:stopY,:]


def pad_image_x(img, normX):
    """
    Pads image along X direction (balanced)
    """

    diff = normX - img.shape[1]
    padWidth = diff//2

    padRightWidth = padWidth
    if diff % 2 != 0:
        padLeftWidth = padWidth+1
    else:
        padLeftWidth = padWidth

    return np.concatenate(
        (
            np.zeros((img.shape[0], padLeftWidth, img.shape[2])),
            img,
            np.zeros((img.shape[0], padRightWidth, img.shape[2]))
        ),
        axis=1
    )


def pad_image_y(img, normY):
    """
    Pads image along Y direction (balanced)
    """

    diff = normY - img.shape[0]
    padWidth = diff//2

    padBottomWidth = padWidth
    if diff % 2 != 0:
        padTopWidth = padWidth+1
    else:
        padTopWidth = padWidth

    return np.concatenate(
        (
            np.zeros((padTopWidth, img.shape[1], img.shape[2])),
            img,
            np.zeros((padBottomWidth, img.shape[1], img.shape[2]))
        ),
        axis=0
    )


def normalize_images(images, normX, normY):
    """
    process the images to fit new size
    if original size is greater than norm -> crop image by center
    if original size is less than norm -> padd image with zeros
    regardless enforce RGBA representation
    """

    for imgName in images.keys():
        print(imgName)
        try:
            img = images[imgName]['img']
            img = enforce_RGBA_channels(img)

            if img.shape[1] > normX: # crop
                img = crop_image_x(img,normX)
            elif img.shape[1] < normX: # pad
                img = pad_image_x(img,normX)

            if img.shape[0] > normY: # crop
                img = crop_image_y(img,normY)
            elif img.shape[0] < normY: # pad
                img = pad_image_y(img,normY)

            images[imgName]['img'] = img

        except Exception as e:
            print(e)
            traceback.print_exc()

    return images


def package(images):
    """
    Given image dictionary generate numpy arrays of images and labels
    """

    labelList = []

    first = True
    for key in images.keys():
        if first:
            print(images[key]['img'].shape)
            imgSet = images[key]['img']
            first = False
        else:
            imgSet = np.concatenate((imgSet,images[key]['img']),axis=0)

        labelList.append(images[key]['lbl'])

        print(join(images[key]['dir'],key))
        print(images[key]['lbl'])

    return np.array(imgSet), np.array(labelList)


def save(destFileName, images, labels):
    """
    Given storage location and numpy arrays, store in file
    """
    np.save(destFileName + '_image.npy',images)
    np.save(destFileName + '_label.npy',labels)


if __name__ == '__main__':
    for set in ['Train','Test','Validate']:
        imageDct = extract('./data/raw/{}'.format(set),{},'',5)
        imageDct = preprocessing(imageDct)
        #images, labels = package(imageDct)
        #save('./data/{}'.format(set),images,labels)
