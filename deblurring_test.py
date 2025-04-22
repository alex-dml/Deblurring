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


dataset_dir = "E:/Datasets/jpg/"

# Utiliser tf.data pour charger et traiter les images
image_paths = os.listdir(dataset_dir)
img_size = 256

BATCH_SIZE = 16
EPOCHS = 200
SHAPE = (BATCH_SIZE, img_size, img_size, 3)


indices = np.arange(len(image_paths))
train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
train_idx, val_idx = train_test_split(train_idx, test_size=0.2, random_state=42)

selected_train_paths = [image_paths[i] for i in train_idx]
selected_val_paths = [image_paths[i] for i in val_idx]
selected_test_paths = [image_paths[i] for i in test_idx]

train_gen = ImageGenerator(selected_train_paths, dataset_dir, BATCH_SIZE, SHAPE, transformation=False)
val_gen = ImageGenerator(selected_val_paths, dataset_dir, BATCH_SIZE, SHAPE, transformation=False)
test_gen = ImageGenerator(selected_test_paths, dataset_dir, BATCH_SIZE, SHAPE, transformation=False)

# Construire le modèle
model = build_autoencoder(img_size)
opt = tf.keras.optimizers.Adam(lr = 0.0002)

# opt = tf.keras.callbacks.ReduceLROnPlateau(
#     monitor='val_loss',
#     factor=0.5,
#     patience=5,
#     min_lr=0.00001,

# )
# Load model
model.load_weights('D:/checkpoints_deblurring/res_ok_0503/')
model.compile(optimizer=opt, loss=total_loss, metrics=ssim_metric)

model.summary()
checkpoint_path= "D:/checkpoints_deblurring/check.{epoch:03d}"

check_callback = [tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True),
                  tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15)]

# model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS, callbacks=check_callback)

# Affichage des résultats
restored_images = model.predict(test_gen)
cv2.imshow("restored", restored_images[0])
cv2.imshow("test", test_gen[0][0][0])
cv2.imshow("gt", test_gen[0][1][0])
cv2.waitKey()

