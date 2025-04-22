# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 15:25:43 2025

@author: duminil
"""
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Conv2D, Multiply, Add, Concatenate

def attention_model(inp, filt):
    
    channels = inp.shape[-1]
    
    conv = Conv2D(channels, (3,3), activation="relu")(inp)
    conv = Conv2D(channels, (3,3), activation="relu")(conv)
    # conv = tf.keras.layers.LayerNormalization()(conv)
    
    avgp = tf.reduce_mean(conv, axis=[1,2], keepdims=True)
    maxp = tf.reduce_max(conv, axis=[1,2], keepdims=True)

    # mlp
    maxp = layers.Dense(3)(maxp)
    avgp = layers.Dense(3)(avgp)
    # Concat
    concat = tf.concat([maxp, avgp], axis=-1)
    
    attention_map = Conv2D(filters=channels, kernel_size=7, padding='same', activation='sigmoid')(concat)
    attention_map = tf.image.resize(attention_map, (inp.shape[1], inp.shape[2]))
    
    end_channel = layers.multiply([attention_map, inp])
    # =========================================================================
    #     Spatial attention
    # =========================================================================
    maxp2 = tf.reduce_max(end_channel, axis=[-1], keepdims=True)
    avgp2 = tf.reduce_mean(end_channel, axis=[-1], keepdims=True)
    
    concat2 = tf.concat([maxp2, avgp2], axis=-1)
    
    attention_map2 = Conv2D(filters=channels, kernel_size=7, padding='same')(concat2)
    end_spatial = layers.multiply([attention_map2, end_channel])
    
    conv = Conv2D(filt, (3,3))(end_spatial)
    out = tf.keras.activations.sigmoid(conv)
    
    return out

class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self, input_dim):
        super(AttentionLayer, self).__init__()

    # def build(self, filter_):
        self.cnn = Conv2D(input_dim, (1,1), padding='same')
        
    def call(self, features):
        c0 = features
        c1 = self.cnn(c0)
        softmax = tf.keras.layers.Softmax()(c1)
        
        c2 = Multiply()([softmax, c0])
        c2 = self.cnn(c2)
        c2 = tf.keras.layers.LayerNormalization()(c2)
        c2 = tf.keras.layers.ReLU()(c2)
        c2 = self.cnn(c2)
        
        output = Add()([c0, c2])
        
        return output

def multiscale_module(inp):
    
    conv1 = tf.keras.layers.Conv2D(filters=3, kernel_size=3, padding='same', activation='relu')(inp)
    conv2 = tf.keras.layers.Conv2D(filters=3, kernel_size=5, padding='same', activation='relu')(inp)
    conv3 = tf.keras.layers.Conv2D(filters=3, kernel_size=7, padding='same', activation='relu')(inp)
    concat = tf.concat([conv1, conv2, conv3], axis=-1)
    
    return concat

def resnet(filters, inputs):
    
    x = tf.keras.layers.Conv2D(filters, (3, 3), activation="relu", padding="same")(inputs)
    x = tf.keras.layers.Conv2D(filters, (3, 3), activation="relu", padding="same")(x)
    connection = tf.concat([inputs, x], axis=-1)
    
    return connection
    
    
    
# Création du modèle autoencodeur
def build_autoencoder(img_size):
    input_layer = tf.keras.layers.Input(shape=(img_size, img_size, 3))
    
    conv1 = tf.keras.layers.Conv2D(filters=3, kernel_size=3, activation="relu")(input_layer)
    m = multiscale_module(conv1)

    # Encodeur
    x1 = tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same")(m)
    x1 = tf.keras.layers.BatchNormalization()(x1)
    x1 = tf.keras.layers.MaxPooling2D((2, 2))(x1)
    r = resnet(64, x1)

    x2 = tf.keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same")(r)
    x2 = tf.keras.layers.BatchNormalization()(x2)
    x2 = tf.keras.layers.MaxPooling2D((2, 2))(x2)
    r = resnet(128, x2)
    
    x3 = tf.keras.layers.Conv2D(256, (3, 3), activation="relu", padding="same")(r)
    x3 = tf.keras.layers.BatchNormalization()(x3)
    x3 = tf.keras.layers.MaxPooling2D((2, 2))(x3)
    
    r = resnet(256, x3)
    # r = resnet(256, r)
    
    # Décodeur
  
    d1 = tf.keras.layers.Conv2DTranspose(256, (3, 3), activation="relu", padding="same")(r)
    skip_attention = attention_model(x3, 256)
    skip_attention = tf.image.resize(skip_attention, (d1.shape[1], d1.shape[2]))
    print('skip_attention', skip_attention.shape)
    x = Concatenate()([d1, skip_attention])
    d1 = tf.keras.layers.BatchNormalization()(x)
    d1 = tf.keras.layers.UpSampling2D((2, 2))(d1)
    r = resnet(256, d1)

    d2 = tf.keras.layers.Conv2DTranspose(128, (3, 3), activation="relu", padding="same")(r)
    skip_attention = attention_model(x2, 128)
    skip_attention = tf.image.resize(skip_attention, (d2.shape[1], d2.shape[2]))
    x = Concatenate()([d2, skip_attention])
    d2 = tf.keras.layers.BatchNormalization()(x)
    d2 = tf.keras.layers.UpSampling2D((2, 2))(d2)
    r = resnet(128, d2)

    d3 = tf.keras.layers.Conv2DTranspose(64, (3, 3), activation="relu", padding="same")(r)
    skip_attention = attention_model(x1, 64)
    skip_attention = tf.image.resize(skip_attention, (d3.shape[1], d3.shape[2]))
    x = Concatenate()([d3, skip_attention])
    d3 = tf.keras.layers.BatchNormalization()(x)
    d3 = tf.keras.layers.UpSampling2D((2, 2))(d3)
    r = resnet(64, d3)
    
    output_layer = tf.keras.layers.Conv2D(3, (3, 3), activation="sigmoid", padding="same")(r)
    output_layer = tf.image.resize(output_layer, (256, 256))
    
    model = tf.keras.models.Model(input_layer, output_layer)
    return model