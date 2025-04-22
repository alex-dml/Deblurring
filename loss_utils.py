# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 15:26:53 2025

@author: duminil

"""
import tensorflow as tf

def ssim_metric(y_true, y_pred):
    return tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0))

def ssim_loss(y_true, y_pred):
    return tf.reduce_mean(1 - tf.image.ssim(y_true, y_pred, 1.0))

def pix_loss(y_true, y_pred):
    N = (1/(256*256))
    
    return  N * tf.reduce_sum(tf.math.pow((y_pred - y_true),2))

def edge_penalty(y_true, y_pred):
    
    return tf.reduce_mean(tf.image.sobel_edges(y_pred))

def total_loss(y_true, y_pred):
    
    return edge_penalty(y_true, y_pred) + pix_loss(y_true, y_pred)
    # return 0.8 * edge_penalty(y_true, y_pred) + 1.2 * pix_loss(y_true, y_pred)