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
        images, labels = package(imageDct)
        save('./data/{}'.format(set),images,labels)
