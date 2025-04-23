# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 10:54:34 2025

@author: duminil
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os, cv2
from sklearn.model_selection import train_test_split
from data import ImageGenerator
from models import build_autoencoder
from loss_utils import total_loss, ssim_metric

BATCH_SIZE = 16
EPOCHS = 200
IMG_SIZE = 256
SHAPE = (BATCH_SIZE, IMG_SIZE, IMG_SIZE, 3)

dataset_dir = "E:/Datasets/jpg/"
image_paths = os.listdir(dataset_dir)



model = build_autoencoder(IMG_SIZE)

model.load_weights('D:/checkpoints_deblurring/res_ok_0503/')
opt = tf.keras.optimizers.Adam(lr = 0.0002)
model.compile(optimizer=opt, loss=total_loss, metrics=ssim_metric)

test_idx=[]
selected_test_paths = [image_paths[i] for i in test_idx]
test_gen = ImageGenerator(selected_test_paths, dataset_dir, BATCH_SIZE, SHAPE, transformation=False)
# Affichage des résultats
restored_images = model.predict(test_gen)
cv2.imshow("restored", restored_images[0])
cv2.imshow("test", test_gen[0][0][0])
cv2.imshow("gt", test_gen[0][1][0])
cv2.waitKey()
