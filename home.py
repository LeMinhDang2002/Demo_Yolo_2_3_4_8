########## Yolo ##########
from keras.models import Model
from keras.layers import Input, Conv2D, concatenate, ZeroPadding2D, LeakyReLU, BatchNormalization, concatenate, MaxPooling2D, UpSampling2D, Concatenate, Layer, Add, LeakyReLU
from keras.initializers import RandomNormal
from keras.regularizers import l2
import tensorflow as tf
from keras.metrics import binary_accuracy
from utils.kmeans import kmeans, iou_dist, euclidean_dist
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler, EarlyStopping
from keras import backend as K
from keras.activations import softplus, tanh
from sklearn.model_selection import train_test_split
import imutils
import cv2
from utils.tools import get_class_weight
optimizer = Adam(learning_rate=5e-5)
import numpy as np
import math
from collections.abc import Iterable

from utils import tools

from functools import wraps
from functools import reduce

class_names = ['motobike']
num_classes = len(class_names)
n_epoch = 10
keep_prob = 0.7
epsilon = 1e-07

optimizer = Adam(learning_rate=1e-4)
callback = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True) 
########## Yolo ##########
### CNN ###
from keras.layers import Flatten, Dense
from keras.applications import ResNetRS420
import os
### CNN ###

### Streamlit ###

import streamlit as st
import time
import pandas as pd
import torch
import tkinter as tk
from tkinter import filedialog

from PIL import Image, ImageDraw
import asyncio

### Streamlit ###


######### Define Class Yolo Data #########
class Yolo_data(object):

    def __init__(self,
                 input_shape=(416, 416, 3),
                 class_names=[]):
        self.input_shape = input_shape
        self.grid_shape = input_shape[0]//32, input_shape[1]//32
        self.class_names = class_names
        self.class_num = len(class_names)
        self.file_names = None
        
    def read_file_to_dataset(
        self, img_path=None, label_path=None,
        label_format="labelimg",
        rescale=1/255,
        preprocessing=None,
        augmenter=None,
        aug_times=1,
        shuffle=True, seed=None,
        encoding="big5",
        thread_num=10):
        
        img_data, label_data, path_list = tools.read_file(
            img_path=img_path, 
            label_path=label_path,
            label_format=label_format,
            size=self.input_shape[:2], 
            grid_shape=self.grid_shape,
            class_names=self.class_names,
            rescale=rescale,
            preprocessing=preprocessing,
            augmenter=augmenter,
            aug_times=aug_times,
            shuffle=shuffle, seed=seed,
            encoding=encoding,
            thread_num=thread_num)
        self.file_names = path_list

        return img_data, label_data
    
    def read_file_to_dataset_x2label(
        self, img_path=None, label_path=None,
        label_format="labelimg",
        rescale=1/255,
        preprocessing=None,
        augmenter=None,
        aug_times=1,
        shuffle=True, seed=None,
        encoding="big5",
        thread_num=10):
        
        grid_amp = 2**(self.fpn_layers - 1)
        grid_shape = (self.grid_shape[0]*grid_amp,
                      self.grid_shape[1]*grid_amp)
        img_data, label_data, path_list = tools.read_file(
            img_path=img_path, 
            label_path=label_path,
            label_format=label_format,
            size=self.input_shape[:2], 
            grid_shape=grid_shape,
            class_names=self.class_names,
            rescale=rescale,
            preprocessing=preprocessing,
            augmenter=augmenter,
            aug_times=aug_times,
            shuffle=shuffle, seed=seed,
            encoding=encoding,
            thread_num=thread_num)
        self.file_names = path_list

        label_list = [label_data]
        for _ in range(self.fpn_layers - 1):
            label_data = tools.down2xlabel(label_data)
            label_list.insert(0, label_data)

        return img_data, label_list
    
######### Define Class Yolo Data #########

######### Function Read Data ############
def ReadData(type = '1_dimension', path = None):
    yolo_data = Yolo_data(class_names=class_names)
    # img_path   = "../01_1K_MNIST/mnist_train/"
    # label_path = "../01_1K_MNIST/xml_train/"
    # img_path   = "D:/Number Plate Region/Mydata/TrainImage/"
    # label_path = "D:/Number Plate Region/Mydata/XMLTrainImage/"
    img_path   = path + '/Image/'
    label_path = path + '/XML/'
    if type == '1_dimension':
        train_img, train_label = yolo_data.read_file_to_dataset(
            img_path, label_path,
            label_format="labelimg",
            thread_num=1,
            shuffle=False)
    else:
        train_img, train_label = yolo_data.read_file_to_dataset_x2label(
            img_path, label_path,
            label_format="labelimg",
            thread_num=1,
            shuffle=False)
    
    return train_img, train_label
######### Function Read Data ############

######### Yolov2 ###############
def _bernoulli(shape, mean):
    return tf.nn.relu(tf.sign(mean - tf.random.uniform(shape, minval=0, maxval=1, dtype=tf.float32)))

class DropBlock2D(tf.keras.layers.Layer):
    def __init__(self, keep_prob, block_size, scale=True,name=None, **kwargs):
        super(DropBlock2D, self).__init__(name="DropBlock2D")
        self.keep_prob = float(keep_prob) if isinstance(keep_prob, int) else keep_prob
        self.block_size = int(block_size)
        self.names = name
        self.scale = tf.constant(scale, dtype=tf.bool) if isinstance(scale, bool) else scale
        super(DropBlock2D, self).__init__(**kwargs)
        
    def get_config(self):
        config = super().get_config().copy()
        config.update( {"block_size": self.block_size,"keep_prob": self.keep_prob,"name": self.names })

        return config

    def compute_output_shape(self, input_shape):
        return input_shape

    def build(self, input_shape):
        assert len(input_shape) == 4
        _, self.h, self.w, self.channel = input_shape.as_list()
        # pad the mask
        p1 = (self.block_size - 1) // 2
        p0 = (self.block_size - 1) - p1
        self.padding = [[0, 0], [p0, p1], [p0, p1], [0, 0]]             
        self.set_keep_prob()
        super(DropBlock2D, self).build(input_shape)

    def call(self, inputs, training=None, **kwargs):
        def drop():
            mask = self._create_mask(tf.shape(inputs))
            ### Apply the mask
            output = inputs * mask
            output = tf.cond(self.scale,
                             ### Normalize the features 
                             true_fn=lambda: output *tf.cast(tf.size(mask), dtype=tf.float32)  / tf.reduce_sum(mask), ### if self.scale == true
                             false_fn=lambda: output)  ### if self.scale == false
            return output

        if training is None:
            training = K.learning_phase()
        # output = tf.cond(tf.logical_or(tf.logical_not(training), tf.equal(self.keep_prob, 1.0)),
        ### If condition is true then output is inputs, false then drop
        output = tf.cond(tf.logical_or(tf.logical_not(True), tf.equal(self.keep_prob, 1.0)),
                         true_fn=lambda: inputs, #### if tf.logical_or(tf.logical_not(True), tf.equal(self.keep_prob, 1.0)) == True
                         false_fn=drop)          #### if tf.logical_or(tf.logical_not(True), tf.equal(self.keep_prob, 1.0)) == False
        return output

    def set_keep_prob(self, keep_prob=None):
        """This method only supports Eager Execution"""
        if keep_prob is not None:
            self.keep_prob = keep_prob
        w, h = tf.cast(self.w, dtype=tf.float32), tf.cast(self.h, dtype=tf.float32)
        ### init param gamma
        self.gamma = (1. - self.keep_prob) * (w * h) / (self.block_size ** 2) / \
                     ((w - self.block_size + 1) * (h - self.block_size + 1))

    def _create_mask(self, input_shape):
        sampling_mask_shape = tf.stack([input_shape[0],
                                       self.h - self.block_size + 1,
                                       self.w - self.block_size + 1,
                                       self.channel])
        mask = _bernoulli(sampling_mask_shape, self.gamma)
        mask = tf.pad(mask, self.padding)
        mask = tf.nn.max_pool(mask, [1, self.block_size, self.block_size, 1], [1, 1, 1, 1], 'SAME')
        mask = 1 - mask
        return mask

class Yolov2(object):

    def __init__(self,
                 input_shape=(416, 416, 3),
                 class_names=[]):
        self.input_shape = input_shape
        self.grid_shape = input_shape[0]//32, input_shape[1]//32
        self.bbox_num = 5
        self.class_names = class_names
        self.class_num = len(class_names)
        self.anchors = None
        self.model = None
        self.file_names = None

    '''
    Leaky Convolutional
    '''
    def Conv2D_BN_Leaky(self, input_tensor, *args):
        output_tensor = Conv2D(*args, 
                            padding='same',      
                            kernel_initializer='he_normal')(input_tensor)
        output_tensor = BatchNormalization()(output_tensor)
        output_tensor = LeakyReLU(alpha=0.1)(output_tensor)
        return output_tensor
    

    '''
    Backbone
    '''
    def Backbone_darknet(self,input_tensor):
        conv1 = self.Conv2D_BN_Leaky(input_tensor, 32, 3) ### 32 is number filters, 3 is size of filter
        conv1 = DropBlock2D(keep_prob=keep_prob, block_size=3)(conv1) ### function call() will be run
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = self.Conv2D_BN_Leaky(pool1, 64, 3)
        conv2 = DropBlock2D(keep_prob=keep_prob, block_size=3)(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = self.Conv2D_BN_Leaky(pool2, 128, 3)
        conv3 = DropBlock2D(keep_prob=keep_prob, block_size=3)(conv3)
        conv3 = self.Conv2D_BN_Leaky(conv3, 64, 1)
        conv3 = DropBlock2D(keep_prob=keep_prob, block_size=3)(conv3)
        conv3 = self.Conv2D_BN_Leaky(conv3, 128, 3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        conv3 = DropBlock2D(keep_prob=keep_prob, block_size=3)(conv3)
        
        conv4 = self.Conv2D_BN_Leaky(pool3, 256, 3)
        conv4 = DropBlock2D(keep_prob=keep_prob, block_size=3)(conv4)
        conv4 = self.Conv2D_BN_Leaky(conv4, 128, 1)
        conv4 = DropBlock2D(keep_prob=keep_prob, block_size=3)(conv4)
        conv4 = self.Conv2D_BN_Leaky(conv4, 256, 3)
        conv4 = DropBlock2D(keep_prob=keep_prob, block_size=3)(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

        conv5 = self.Conv2D_BN_Leaky(pool4, 512, 3)
        conv5 = DropBlock2D(keep_prob=keep_prob, block_size=3)(conv5)
        conv5 = self.Conv2D_BN_Leaky(conv5, 256, 1)
        conv5 = DropBlock2D(keep_prob=keep_prob, block_size=3)(conv5)
        conv5 = self.Conv2D_BN_Leaky(conv5, 512, 3)
        conv5 = DropBlock2D(keep_prob=keep_prob, block_size=3)(conv5)
        conv5 = self.Conv2D_BN_Leaky(conv5, 256, 1)
        conv5 = DropBlock2D(keep_prob=keep_prob, block_size=3)(conv5)
        conv5 = self.Conv2D_BN_Leaky(conv5, 512, 3)
        conv5 = DropBlock2D(keep_prob=keep_prob, block_size=3)(conv5)
        pool5 = MaxPooling2D(pool_size=(2, 2))(conv5)
        
        conv6 = self.Conv2D_BN_Leaky(pool5, 1024, 3)
        conv6 = DropBlock2D(keep_prob=keep_prob, block_size=3)(conv6)
        conv6 = self.Conv2D_BN_Leaky(conv6, 512, 1)
        conv6 = DropBlock2D(keep_prob=keep_prob, block_size=3)(conv6)
        conv6 = self.Conv2D_BN_Leaky(conv6, 1024, 3)
        conv6 = DropBlock2D(keep_prob=keep_prob, block_size=3)(conv6)
        conv6 = self.Conv2D_BN_Leaky(conv6, 512, 1)
        conv6 = DropBlock2D(keep_prob=keep_prob, block_size=3)(conv6)
        
        output_tensor = self.Conv2D_BN_Leaky(conv6, 1024, 3)
        
        return output_tensor

    '''
    Yolo Neck
    '''
    def yolo_neck(self, input_shape=(416, 416, 3),
              backbone="darknet",
              pretrained_darknet=None):
        inputs = Input(input_shape)
        ### input is inputs and output is Backbone_darknet 
        darknet = Model(inputs, self.Backbone_darknet(inputs))
        if pretrained_darknet is not None:
            darknet.set_weights(pretrained_darknet.get_weights())
        
        ### Get class 43th of model darknet for passthrough
        passthrough = darknet.layers[43].output
        conv = self.Conv2D_BN_Leaky(darknet.output, 1024, 3)
        conv = DropBlock2D(keep_prob=keep_prob, block_size=3)(conv)
        conv = self.Conv2D_BN_Leaky(conv, 1024, 3)   

        passthrough = self.Conv2D_BN_Leaky(passthrough, 512, 3)
        passthrough = tf.nn.space_to_depth(passthrough, 2)

        merge = concatenate([passthrough, conv], axis=-1)

        outputs = self.Conv2D_BN_Leaky(merge, 1024, 3)

        model = Model(inputs, outputs)
        
        return model
    
    '''
    Yolo Head
    '''
    def yolo_head(self, model_body, class_num=10, 
              anchors=[(0.04405615, 0.05210654),
                       (0.14418923, 0.15865615),
                       (0.25680231, 0.42110308),
                       (0.60637077, 0.27136769),
                       (0.75157846, 0.70525231)]):
        anchors = np.array(anchors)
        inputs = model_body.input
        output = model_body.output
        output_list = []
        for box in anchors:
            xy_output = Conv2D(2, 1,
                            padding='same',
                            activation='sigmoid',
                            kernel_initializer='he_normal')(output)
            wh_output = Conv2D(2, 1,
                            padding='same',
                            activation='exponential',
                            kernel_initializer='he_normal')(output)
            wh_output = wh_output * box
            c_output = Conv2D(1, 1,
                            padding='same',
                            activation='sigmoid',
                            kernel_initializer='he_normal')(output)
            p_output = Conv2D(class_num, 1,
                            padding = 'same',
                            activation='softmax',
                            kernel_initializer='he_normal')(output)
            output_list += [xy_output,
                            wh_output,
                            c_output,
                            p_output]

        outputs = concatenate(output_list, axis=-1)
        
        model = Model(inputs, outputs)

        return model
    
    '''
    IoU
    '''
    def cal_iou(self, xywh_true, xywh_pred, grid_shape):
        grid_shape = np.array(grid_shape[::-1])
        xy_true = xywh_true[..., 0:2]/grid_shape # N*S*S*1*2
        wh_true = xywh_true[..., 2:4]

        xy_pred = xywh_pred[..., 0:2]/grid_shape # N*S*S*B*2
        wh_pred = xywh_pred[..., 2:4]
        
        half_xy_true = wh_true / 2.
        mins_true    = xy_true - half_xy_true
        maxes_true   = xy_true + half_xy_true

        half_xy_pred = wh_pred / 2.
        mins_pred    = xy_pred - half_xy_pred
        maxes_pred   = xy_pred + half_xy_pred       
        
        intersect_mins  = tf.maximum(mins_pred,  mins_true)
        intersect_maxes = tf.minimum(maxes_pred, maxes_true)
        intersect_wh    = tf.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]
        
        true_areas = wh_true[..., 0] * wh_true[..., 1]
        pred_areas = wh_pred[..., 0] * wh_pred[..., 1]

        union_areas = pred_areas + true_areas - intersect_areas
        iou_scores  = (intersect_areas + epsilon)/(union_areas + epsilon)
        
        return iou_scores
    

    '''
    Yolo Loss Function
    '''
    def wrap_yolo_loss(self, grid_shape,
                   bbox_num,
                   class_num,
                   anchors,
                   binary_weight=1,
                   loss_weight=[1, 1, 1, 1],
                   ignore_thresh=.6,
                   ):
        def yolo_loss(y_true, y_pred):
            panchors = tf.reshape(anchors, (1, 1, 1, bbox_num, 2))

            y_true = tf.reshape(
                y_true,
                (-1, *grid_shape, 1, 5 + class_num)) # N*S*S*1*(5+C)
            y_pred = tf.reshape(
                y_pred,
                (-1, *grid_shape, bbox_num, 5 + class_num)) # N*S*S*B*(5+C)

            xywh_true = y_true[..., :4] # N*S*S*1*4
            xywh_pred = y_pred[..., :4] # N*S*S*B*4

            iou_scores = self.cal_iou(xywh_true, xywh_pred, grid_shape) # N*S*S*B
            
            response_mask = tf.one_hot(tf.argmax(iou_scores, axis=-1),
                                    depth=bbox_num,
                                    dtype=xywh_true.dtype) # N*S*S*B

            has_obj_mask = y_true[..., 4]*response_mask # N*S*S*B
            has_obj_mask_exp = tf.expand_dims(has_obj_mask, axis=-1) # N*S*S*B*1

            no_obj_mask = tf.cast(
                iou_scores < ignore_thresh,
                iou_scores.dtype) # N*S*S*B
            no_obj_mask = (1 - has_obj_mask)*no_obj_mask # N*S*S*B

            xy_true = y_true[..., 0:2] # N*S*S*1*2
            xy_pred = y_pred[..., 0:2] # N*S*S*B*2

            wh_true = tf.maximum(y_true[..., 2:4]/panchors, epsilon) # N*S*S*1*2
            wh_pred = y_pred[..., 2:4]/panchors
            
            wh_true = tf.math.log(wh_true) # N*S*S*B*2
            wh_pred = tf.math.log(wh_pred) # N*S*S*B*2

            c_pred = y_pred[..., 4] # N*S*S*B

            box_loss_scale = 2 - y_true[..., 2:3]*y_true[..., 3:4] # N*S*S*1*1

            xy_loss = tf.reduce_sum(
                tf.reduce_mean(
                    has_obj_mask_exp # N*S*S*B*1
                    *box_loss_scale # N*S*S*1*1
                    *tf.square(xy_true - xy_pred), # N*S*S*B*2
                    axis=0))

            wh_loss = tf.reduce_sum(
                tf.reduce_mean(
                    has_obj_mask_exp # N*S*S*B*1
                    *box_loss_scale # N*S*S*1*1
                    *tf.square(wh_true - wh_pred), # N*S*S*B*2
                    axis=0))

            has_obj_c_loss = tf.reduce_sum(
                    tf.reduce_mean(
                    has_obj_mask # N*S*S*B
                    *(tf.square(1 - c_pred)), # N*S*S*B
                    axis=0))

            no_obj_c_loss = tf.reduce_sum(
                    tf.reduce_mean(
                    no_obj_mask # N*S*S*1
                    *(tf.square(0 - c_pred)), # N*S*S*B
                    axis=0))
            
            c_loss = has_obj_c_loss + binary_weight*no_obj_c_loss

            p_true = y_true[..., -class_num:] # N*S*S*1*C
            p_pred = y_pred[..., -class_num:] # N*S*S*B*C
            p_pred = tf.clip_by_value(p_pred, epsilon, 1 - epsilon)
            p_loss = -tf.reduce_sum(
                tf.reduce_mean(
                    has_obj_mask_exp # N*S*S*B*1
                    *(p_true*tf.math.log(p_pred)
                    + (1 - p_true)*tf.math.log(1 - p_pred)), # N*S*S*B*C
                    axis=0))

            loss = (loss_weight[0]*xy_loss
                    + loss_weight[1]*wh_loss
                    + loss_weight[2]*c_loss
                    + loss_weight[3]*p_loss)

            return loss

        return yolo_loss
    
    '''
    Object Accuracy
    '''
    def wrap_obj_acc(self, grid_shape, bbox_num, class_num):
        def obj_acc(y_true, y_pred):
            y_true = tf.reshape(
                y_true,
                (-1, *grid_shape, 1, 5 + class_num)) # N*S*S*1*5+C
            y_pred = tf.reshape(
                y_pred,
                (-1, *grid_shape, bbox_num, 5 + class_num)) # N*S*S*B*5+C
            
            c_true = y_true[..., 4] # N*S*S*1
            c_pred = tf.reduce_max(y_pred[..., 4], # N*S*S*B
                                axis=-1,
                                keepdims=True) # N*S*S*1

            bi_acc = binary_accuracy(c_true, c_pred)

            return bi_acc
        return obj_acc
    
    '''
    MeanIoU
    '''
    def wrap_mean_iou(self, grid_shape, bbox_num, class_num):
        def mean_iou(y_true, y_pred):
            y_true = tf.reshape(
                y_true,
                (-1, *grid_shape, 1, 5 + class_num)) # N*S*S*1*5+C
            y_pred = tf.reshape(
                y_pred,
                (-1, *grid_shape, bbox_num, 5 + class_num)) # N*S*S*B*5+C

            has_obj_mask = y_true[..., 4] # N*S*S*1
            
            xywh_true = y_true[..., :4] # N*S*S*1*4
            xywh_pred = y_pred[..., :4] # N*S*S*B*4

            iou_scores = self.cal_iou(xywh_true, xywh_pred, grid_shape) # N*S*S*B
            iou_scores = tf.reduce_max(iou_scores, axis=-1, keepdims=True) # N*S*S*1
            iou_scores = iou_scores*has_obj_mask # N*S*S*B

            num_p = tf.reduce_sum(has_obj_mask)

            return tf.reduce_sum(iou_scores)/(num_p + epsilon)
        return mean_iou

    '''
    Class Accuracy
    '''
    def wrap_class_acc(self, grid_shape, bbox_num, class_num):
        def class_acc(y_true, y_pred):
            y_true = tf.reshape(
                y_true,
                (-1, *grid_shape, 1, 5 + class_num)) # N*S*S*1*5+C
            y_pred = tf.reshape(
                y_pred,
                (-1, *grid_shape, bbox_num, 5 + class_num)) # N*S*S*B*5+C

            has_obj_mask = y_true[..., 4] # N*S*S*1

            pi_true = tf.argmax(y_true[..., -class_num:], # N*S*S*1*C
                                axis=-1) # N*S*S*1
            pi_pred = tf.argmax(y_pred[..., -class_num:], # N*S*S*B*C
                                axis=-1) # N*S*S*B
            
            equal_mask = tf.cast(pi_true == pi_pred,
                                dtype=y_true.dtype) # N*S*S*B
            equal_mask = equal_mask*has_obj_mask # N*S*S*B

            num_p = tf.reduce_sum(has_obj_mask)*bbox_num

            return tf.reduce_sum(equal_mask)/(num_p + epsilon)
        return class_acc
    
    ''''''



    '''
    Model Create
    '''
    def create_model(self,
                     anchors=[[0.75157846, 0.70525231],
                              [0.60637077, 0.27136769],
                              [0.25680231, 0.42110308],
                              [0.14418923, 0.15865615],
                              [0.04405615, 0.05210654]],
                     backbone="darknet",
                     pretrained_weights=None,
                     pretrained_darknet=None):
        
        model_body = self.yolo_neck(self.input_shape,
                               backbone,
                               pretrained_darknet)

        self.model = self.yolo_head(model_body,
                               self.class_num,
                               anchors)
         
        if pretrained_weights is not None:
            self.model.load_weights(pretrained_weights)
        self.anchors = anchors
        self.grid_shape = self.model.output.shape[1:3]
        self.bbox_num = len(anchors)

    '''
    Loss Create
    '''
    def loss(self,
             binary_weight=1,
             loss_weight=[1, 1, 5, 1],
             ignore_thresh=0.6):
        
        if isinstance(loss_weight, dict):
            loss_weight_list = []
            loss_weight_list.append(loss_weight["xy"])
            loss_weight_list.append(loss_weight["wh"])
            loss_weight_list.append(loss_weight["conf"])
            loss_weight_list.append(loss_weight["prob"])
            loss_weight = loss_weight_list
        
        return self.wrap_yolo_loss(
            grid_shape=self.grid_shape,
            bbox_num=self.bbox_num, 
            class_num=self.class_num,
            anchors=self.anchors,
            binary_weight=binary_weight,
            loss_weight=loss_weight,
            ignore_thresh=ignore_thresh,
            )
    
    '''
    Metrics Create
    '''
    def metrics(self, type="obj_acc"):
        
        metrics_list = []     
        if "obj" in type:
            metrics_list.append(
                self.wrap_obj_acc(
                    self.grid_shape, 
                    self.bbox_num, 
                    self.class_num))
        if "iou" in type:
            metrics_list.append(
                self.wrap_mean_iou(
                    self.grid_shape, 
                    self.bbox_num, 
                    self.class_num))
        if "class" in type:
            metrics_list.append(
                self.wrap_class_acc(
                    self.grid_shape, 
                    self.bbox_num, 
                    self.class_num))
        
        return metrics_list
    
######### Yolov2 ###############
######### Build Model Yolov2 ##########
def Model_Yolov2(train_img, train_label, val_img, val_label, class_names, batch_size, epochs, anchors):
    yolo = Yolov2(class_names=class_names)
    yolo.create_model(anchors=anchors)

    binary_weight = get_class_weight(
    train_label[..., 4:5],
    method='binary'
    )
    # print(binary_weight)

    loss_weight = {
        "xy":1,
        "wh":1,
        "conf":5,
        "prob":1
        }

    loss_fn = yolo.loss(
        binary_weight=binary_weight,
        loss_weight=loss_weight
        )
    metrics = yolo.metrics("obj+iou+class")

    yolo.model.compile(
        optimizer = optimizer,
        #optimizer=SGD(learning_rate=1e-10, momentum=0.9, decay=5e-4),
        loss = loss_fn,
        metrics = metrics
        )
    
    train_history = yolo.model.fit(train_img,
                                    train_label,
                                    epochs = epochs,
                                    batch_size=batch_size,
                                    verbose=1,
                                    validation_data=(val_img, val_label),
                                    callbacks=[callback]
                                    )
    return yolo
######### Build Model Yolov2 ##########


######### Yolov3 #############
def mish(x):
    return x * tf.math.tanh(tf.math.softplus(x))
def upsample(input_layer):
    return tf.image.resize(input_layer, (input_layer.shape[1] * 2, input_layer.shape[2] * 2), method='nearest')
class Yolov3(object):

    def __init__(self,
                 input_shape=(416, 416, 3),
                 class_names=[]):
        self.input_shape = input_shape
        self.grid_shape = input_shape[0]//32, input_shape[1]//32
        self.abox_num = 3
        self.class_names = class_names
        self.class_num = len(class_names)
        self.fpn_layers = 3
        self.anchors = None
        self.model = None
        self.file_names = None
    
    @wraps(Conv2D)
    def DarknetConv2D(self, *args, **kwargs):
        '''Wrapper to set Darknet parameters for Convolution2D.'''
        darknet_conv_kwargs = {'kernel_initializer': 'he_normal'}
        if kwargs.get('strides') == (2, 2):
            darknet_conv_kwargs['padding'] = 'valid'
        else:
            darknet_conv_kwargs['padding'] = 'same'
        darknet_conv_kwargs.update(kwargs)
        return Conv2D(*args, **darknet_conv_kwargs)
    
    '''
    Residual Block
    '''
    def DarknetConv2D_BN_Leaky(self, input_layer, filters_shape, downsample=False, activate=True, bn=True, activate_type='leaky'):
        if downsample:
            input_layer = ZeroPadding2D(((1, 0), (1, 0)))(input_layer)
            padding = 'valid'
            strides = 2
        else:
            strides = 1
            padding = 'same'

        conv = Conv2D(filters=filters_shape[-1], kernel_size = filters_shape[0], strides=strides,
                    padding=padding, use_bias=not bn, kernel_regularizer=l2(0.0005),
                    kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                    bias_initializer=tf.constant_initializer(0.))(input_layer)
        if bn:
            conv = BatchNormalization()(conv)
        if activate == True:
            if activate_type == "leaky":
                conv = LeakyReLU(alpha=0.1)(conv)
            elif activate_type == "mish":
                conv = mish(conv)

        return conv
    
    '''
    DarknetResidual
    '''
    def DarknetResidual(self, input_layer, input_channel, filter_num1, filter_num2, activate_type='leaky'):
        short_cut = input_layer
        conv = self.DarknetConv2D_BN_Leaky(input_layer, filters_shape=(1, 1, input_channel, filter_num1), activate_type=activate_type)
        conv = self.DarknetConv2D_BN_Leaky(conv       , filters_shape=(3, 3, filter_num1,   filter_num2), activate_type=activate_type)

        residual_output = short_cut + conv
        return residual_output
    
    '''Backbone darknet'''
    def Backbone_darknet(self, input_tensor):
        input_tensor = self.DarknetConv2D_BN_Leaky(input_tensor, (3, 3,  3,  32))
        input_tensor = self.DarknetConv2D_BN_Leaky(input_tensor, (3, 3, 32,  64), downsample=True)

        for i in range(1):
            input_tensor = self.DarknetResidual(input_tensor,  64,  32, 64)

        input_tensor = self.DarknetConv2D_BN_Leaky(input_tensor, (3, 3,  64, 128), downsample=True)

        for i in range(2):
            input_tensor = self.DarknetResidual(input_tensor, 128,  64, 128)

        input_tensor = self.DarknetConv2D_BN_Leaky(input_tensor, (3, 3, 128, 256), downsample=True)

        for i in range(8):
            input_tensor = self.DarknetResidual(input_tensor, 256, 128, 256)

        route_1 = input_tensor
        input_tensor = self.DarknetConv2D_BN_Leaky(input_tensor, (3, 3, 256, 512), downsample=True)

        for i in range(8):
            input_tensor = self.DarknetResidual(input_tensor, 512, 256, 512)

        route_2 = input_tensor
        input_tensor = self.DarknetConv2D_BN_Leaky(input_tensor, (3, 3, 512, 1024), downsample=True)

        for i in range(4):
            input_tensor = self.DarknetResidual(input_tensor, 1024, 512, 1024)

        return route_1, route_2, input_tensor
    
    '''
    Yolo Neck
    '''
    def yolo_neck(self, input_shape=(416, 416, 3),
              pretrained_darknet=None,
              pretrained_weights=None):
        '''Create YOLO_V3 model CNN body in Keras.'''
        input_tensor = Input(input_shape)
        route_1, route_2, conv = self.Backbone_darknet(input_tensor)
        # if pretrained_darknet is not None:
        #     darknet.set_weights(pretrained_darknet.get_weights())
        
        # conv, conv_lobj_branch = make_last_layers(conv, 512)
        conv = self.DarknetConv2D_BN_Leaky(conv, (1, 1, 1024,  512))
        conv = self.DarknetConv2D_BN_Leaky(conv, (3, 3,  512, 1024))
        conv = self.DarknetConv2D_BN_Leaky(conv, (1, 1, 1024,  512))
        conv = self.DarknetConv2D_BN_Leaky(conv, (3, 3,  512, 1024))
        conv = self.DarknetConv2D_BN_Leaky(conv, (1, 1, 1024,  512))
        conv_lobj_branch = self.DarknetConv2D_BN_Leaky(conv, (3, 3, 512, 1024))
        
        conv = self.DarknetConv2D_BN_Leaky(conv, (1, 1, 256, 128))
        conv = upsample(conv)
        
        conv = tf.concat([conv, route_2], axis=-1)
        conv = self.DarknetConv2D_BN_Leaky(conv, (1, 1, 768, 256))
        conv = self.DarknetConv2D_BN_Leaky(conv, (3, 3, 256, 512))
        conv = self.DarknetConv2D_BN_Leaky(conv, (1, 1, 512, 256))
        conv = self.DarknetConv2D_BN_Leaky(conv, (3, 3, 256, 512))
        conv = self.DarknetConv2D_BN_Leaky(conv, (1, 1, 512, 256))
        conv_mobj_branch = self.DarknetConv2D_BN_Leaky(conv, (3, 3, 256, 512))

        conv = self.DarknetConv2D_BN_Leaky(conv, (1, 1, 256, 128))
        conv = upsample(conv)

        conv = tf.concat([conv, route_1], axis=-1)
        conv = self.DarknetConv2D_BN_Leaky(conv, (1, 1, 384, 128))
        conv = self.DarknetConv2D_BN_Leaky(conv, (3, 3, 128, 256))
        conv = self.DarknetConv2D_BN_Leaky(conv, (1, 1, 256, 128))
        conv = self.DarknetConv2D_BN_Leaky(conv, (3, 3, 128, 256))
        conv = self.DarknetConv2D_BN_Leaky(conv, (1, 1, 256, 128))
        conv_sobj_branch = self.DarknetConv2D_BN_Leaky(conv, (3, 3, 128, 256))
        
        model = Model(input_tensor, [conv_lobj_branch, conv_mobj_branch, conv_sobj_branch])

        return model
    
    '''
    Yolo head
    '''
    def yolo_head(self, model_body, class_num=10, 
              anchors=[[0.89663461, 0.78365384],
                       [0.37500000, 0.47596153],
                       [0.27884615, 0.21634615],
                       [0.14182692, 0.28605769],
                       [0.14903846, 0.10817307],
                       [0.07211538, 0.14663461],
                       [0.07932692, 0.05528846],
                       [0.03846153, 0.07211538],
                       [0.02403846, 0.03125000]]):
        anchors = np.array(anchors)
        inputs = model_body.input
        output = model_body.output
        tensor_num = len(output)

        if len(anchors)%tensor_num > 0:
            raise ValueError(("The total number of anchor boxs"
                            " should be a multiple of the number(%s)"
                            " of output tensors") % tensor_num)    
        abox_num = len(anchors)//tensor_num

        outputs_list = []
        for tensor_i, output_tensor in enumerate(output):
            output_list = []
            start_i = tensor_i*abox_num
            for box in anchors[start_i:start_i + abox_num]:
                xy_output = self.DarknetConv2D(2, 1,
                                activation='sigmoid')(output_tensor)
                wh_output = self.DarknetConv2D(2, 1,
                                activation='exponential')(output_tensor)
                wh_output = wh_output * box
                c_output = self.DarknetConv2D(1, 1,
                                activation='sigmoid')(output_tensor)
                p_output = self.DarknetConv2D(class_num, 1,
                                activation='sigmoid')(output_tensor)
                output_list += [xy_output,
                                wh_output,
                                c_output,
                                p_output]

            outputs = concatenate(output_list, axis=-1)
            outputs_list.append(outputs)
        
        model = Model(inputs, outputs_list)    

        return model
    
    '''
    IoU
    '''
    def cal_iou(self, xywh_true, xywh_pred, grid_shape):
        grid_shape = np.array(grid_shape[::-1])
        xy_true = xywh_true[..., 0:2]/grid_shape # N*S*S*1*2
        wh_true = xywh_true[..., 2:4]

        xy_pred = xywh_pred[..., 0:2]/grid_shape # N*S*S*B*2
        wh_pred = xywh_pred[..., 2:4]
        
        half_xy_true = wh_true / 2.
        mins_true    = xy_true - half_xy_true
        maxes_true   = xy_true + half_xy_true

        half_xy_pred = wh_pred / 2.
        mins_pred    = xy_pred - half_xy_pred
        maxes_pred   = xy_pred + half_xy_pred       
        
        intersect_mins  = tf.maximum(mins_pred,  mins_true)
        intersect_maxes = tf.minimum(maxes_pred, maxes_true)
        intersect_wh    = tf.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]
        
        true_areas = wh_true[..., 0] * wh_true[..., 1]
        pred_areas = wh_pred[..., 0] * wh_pred[..., 1]

        union_areas = pred_areas + true_areas - intersect_areas
        iou_scores  = (intersect_areas + epsilon)/(union_areas + epsilon)
        
        return iou_scores
    
    '''
    Yolo Loss
    '''
    def wrap_yolo_loss(self, grid_shape,
                   bbox_num,
                   class_num,
                   anchors,
                   binary_weight=1,
                   loss_weight=[1, 1, 1, 1],
                   ignore_thresh=.6,
                   ):
        def yolo_loss(y_true, y_pred):
            panchors = tf.reshape(anchors, (1, 1, 1, bbox_num, 2))

            y_true = tf.reshape(
                y_true,
                (-1, *grid_shape, 1, 5 + class_num)) # N*S*S*1*(5+C)
            y_pred = tf.reshape(
                y_pred,
                (-1, *grid_shape, bbox_num, 5 + class_num)) # N*S*S*B*(5+C)

            xywh_true = y_true[..., :4] # N*S*S*1*4
            xywh_pred = y_pred[..., :4] # N*S*S*B*4

            iou_scores = self.cal_iou(xywh_true, xywh_pred, grid_shape) # N*S*S*B
            
            response_mask = tf.one_hot(tf.argmax(iou_scores, axis=-1),
                                    depth=bbox_num,
                                    dtype=xywh_true.dtype) # N*S*S*B

            has_obj_mask = y_true[..., 4]*response_mask # N*S*S*B
            has_obj_mask_exp = tf.expand_dims(has_obj_mask, axis=-1) # N*S*S*B*1

            no_obj_mask = tf.cast(
                iou_scores < ignore_thresh,
                iou_scores.dtype) # N*S*S*B
            no_obj_mask = (1 - has_obj_mask)*no_obj_mask # N*S*S*B

            xy_true = y_true[..., 0:2] # N*S*S*1*2
            xy_pred = y_pred[..., 0:2] # N*S*S*B*2

            wh_true = tf.maximum(y_true[..., 2:4]/panchors, epsilon) # N*S*S*1*2
            wh_pred = y_pred[..., 2:4]/panchors
            
            wh_true = tf.math.log(wh_true) # N*S*S*B*2
            wh_pred = tf.math.log(wh_pred) # N*S*S*B*2

            c_pred = y_pred[..., 4] # N*S*S*B

            box_loss_scale = 2 - y_true[..., 2:3]*y_true[..., 3:4] # N*S*S*1*1

            xy_loss = tf.reduce_sum(
                tf.reduce_mean(
                    has_obj_mask_exp # N*S*S*B*1
                    *box_loss_scale # N*S*S*1*1
                    *tf.square(xy_true - xy_pred), # N*S*S*B*2
                    axis=0))

            wh_loss = tf.reduce_sum(
                tf.reduce_mean(
                    has_obj_mask_exp # N*S*S*B*1
                    *box_loss_scale # N*S*S*1*1
                    *tf.square(wh_true - wh_pred), # N*S*S*B*2
                    axis=0))

            has_obj_c_loss = tf.reduce_sum(
                    tf.reduce_mean(
                    has_obj_mask # N*S*S*1
                    *(tf.square(1 - c_pred)), # N*S*S*B
                    axis=0))

            no_obj_c_loss = tf.reduce_sum(
                    tf.reduce_mean(
                    no_obj_mask # N*S*S*1
                    *(tf.square(0 - c_pred)), # N*S*S*B
                    axis=0))
            
            c_loss = has_obj_c_loss + binary_weight*no_obj_c_loss

            p_true = y_true[..., -class_num:] # N*S*S*1*C
            p_pred = y_pred[..., -class_num:] # N*S*S*B*C
            p_pred = tf.clip_by_value(p_pred, epsilon, 1 - epsilon)
            p_loss = -tf.reduce_sum(
                tf.reduce_mean(
                    has_obj_mask_exp # N*S*S*B*1
                    *(p_true*tf.math.log(p_pred)
                    + (1 - p_true)*tf.math.log(1 - p_pred)), # N*S*S*B*C
                    axis=0))
            
            regularizer = tf.reduce_sum(
                tf.reduce_mean(wh_pred**2, axis=0))*0.01

            loss = (loss_weight[0]*xy_loss
                    + loss_weight[1]*wh_loss
                    + loss_weight[2]*c_loss
                    + loss_weight[3]*p_loss
                    + regularizer)

            return loss

        return yolo_loss
    

    '''
    Yolo Accuracy Object
    '''
    def wrap_obj_acc(self, grid_shape, bbox_num, class_num):
        def obj_acc(y_true, y_pred):
            y_true = tf.reshape(
                y_true,
                (-1, *grid_shape, 1, 5 + class_num)) # N*S*S*1*5+C
            y_pred = tf.reshape(
                y_pred,
                (-1, *grid_shape, bbox_num, 5 + class_num)) # N*S*S*B*5+C
            
            c_true = y_true[..., 4] # N*S*S*1
            c_pred = tf.reduce_max(y_pred[..., 4], # N*S*S*B
                                axis=-1,
                                keepdims=True) # N*S*S*1

            bi_acc = binary_accuracy(c_true, c_pred)

            return bi_acc
        return obj_acc
    
    '''Mean IoU'''
    def wrap_mean_iou(self, grid_shape, bbox_num, class_num):
        def mean_iou(y_true, y_pred):
            y_true = tf.reshape(
                y_true,
                (-1, *grid_shape, 1, 5 + class_num)) # N*S*S*1*5+C
            y_pred = tf.reshape(
                y_pred,
                (-1, *grid_shape, bbox_num, 5 + class_num)) # N*S*S*B*5+C

            has_obj_mask = y_true[..., 4] # N*S*S*1
            
            xywh_true = y_true[..., :4] # N*S*S*1*4
            xywh_pred = y_pred[..., :4] # N*S*S*B*4

            iou_scores = self.cal_iou(xywh_true, xywh_pred, grid_shape) # N*S*S*B
            iou_scores = tf.reduce_max(iou_scores, axis=-1, keepdims=True) # N*S*S*1
            iou_scores = iou_scores*has_obj_mask # N*S*S*1

            num_p = tf.reduce_sum(has_obj_mask)

            return tf.reduce_sum(iou_scores)/(num_p + epsilon)
        return mean_iou
    
    '''
    Yolo Accuracy Class
    '''
    def wrap_class_acc(grid_shape, bbox_num, class_num):
        def class_acc(y_true, y_pred):
            y_true = tf.reshape(
                y_true,
                (-1, *grid_shape, 1, 5 + class_num)) # N*S*S*1*5+C
            y_pred = tf.reshape(
                y_pred,
                (-1, *grid_shape, bbox_num, 5 + class_num)) # N*S*S*B*5+C

            has_obj_mask = y_true[..., 4] # N*S*S*1

            pi_true = tf.argmax(y_true[..., -class_num:], # N*S*S*1*C
                                axis=-1) # N*S*S*1
            pi_pred = tf.argmax(y_pred[..., -class_num:], # N*S*S*B*C
                                axis=-1) # N*S*S*B
            
            equal_mask = tf.cast(pi_true == pi_pred,
                                dtype=y_true.dtype) # N*S*S*B
            equal_mask = equal_mask*has_obj_mask # N*S*S*B

            num_p = tf.reduce_sum(has_obj_mask)*bbox_num

            return tf.reduce_sum(equal_mask)/(num_p + epsilon)
        return class_acc
  
    def create_model(self,
                     anchors=[[0.89663461, 0.78365384],
                              [0.37500000, 0.47596153],
                              [0.27884615, 0.21634615],
                              [0.14182692, 0.28605769],
                              [0.14903846, 0.10817307],
                              [0.07211538, 0.14663461],
                              [0.07932692, 0.05528846],
                              [0.03846153, 0.07211538],
                              [0.02403846, 0.03125000]],
                     backbone="full_darknet",
                     pretrained_weights=None,
                     pretrained_darknet="pascal_voc"):
        
        if isinstance(pretrained_darknet, str):
            pre_body_weights = pretrained_darknet
            pretrained_darknet = None
        else:
            pre_body_weights = None
        
        model_body = self.yolo_neck(self.input_shape,
            pretrained_weights=pre_body_weights)

        if pretrained_darknet is not None:
            model_body.set_weights(pretrained_darknet.get_weights())
        self.model = self.yolo_head(model_body,
                               self.class_num,
                               anchors)
         
        if pretrained_weights is not None:
            self.model.load_weights(pretrained_weights)
        self.anchors = anchors
        self.grid_shape = self.model.output[0].shape[1:3]
        self.fpn_layers = len(self.model.output)
        self.abox_num = len(self.anchors)//self.fpn_layers

    def loss(self,
             binary_weight=1,
             loss_weight=[1, 1, 5, 1],
             ignore_thresh=0.6):

        if (not isinstance(binary_weight, Iterable)
            or len(binary_weight) != self.fpn_layers):
            binary_weight = [binary_weight]*self.fpn_layers
        
        if isinstance(loss_weight, dict):
            loss_weight_list = []
            loss_weight_list.append(loss_weight["xy"])
            loss_weight_list.append(loss_weight["wh"])
            loss_weight_list.append(loss_weight["conf"])
            loss_weight_list.append(loss_weight["prob"])
            loss_weight = loss_weight_list
        
        loss_list = []
        for fpn_id in range(self.fpn_layers):
            grid_amp = 2**(fpn_id)
            grid_shape = (self.grid_shape[0]*grid_amp,
                          self.grid_shape[1]*grid_amp)
            anchors_id = self.abox_num*fpn_id
            loss_list.append(self.wrap_yolo_loss(
                grid_shape=grid_shape,
                bbox_num=self.abox_num, 
                class_num=self.class_num,
                anchors=self.anchors[
                    anchors_id:anchors_id + self.abox_num],
                binary_weight=binary_weight[fpn_id],
                loss_weight=loss_weight,
                ignore_thresh=ignore_thresh))
        return loss_list
    
    def metrics(self, type="obj_acc"):
        
        metrics_list = [[] for _ in range(self.fpn_layers)]
        for fpn_id in range(self.fpn_layers):
            grid_amp = 2**(fpn_id)
            grid_shape = (self.grid_shape[0]*grid_amp,
                            self.grid_shape[1]*grid_amp)
            
            if "obj" in type:
                metrics_list[fpn_id].append(
                    self.wrap_obj_acc(
                        grid_shape,
                        self.abox_num, 
                        self.class_num))
            if "iou" in type:
                metrics_list[fpn_id].append(
                    self.wrap_mean_iou(
                        grid_shape,
                        self.abox_num, 
                        self.class_num))
            if "class" in type:
                metrics_list[fpn_id].append(
                    self.wrap_class_acc(
                        grid_shape,
                        self.abox_num, 
                        self.class_num))
        
        return metrics_list
######### Yolov3 #############
######### Build Model Yolov3 ###############
def Model_Yolov3(train_img, train_label, val_img, val_label, class_names, anchors):
    optimizer = Adam(learning_rate=1e-4)
    callback = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True) 

    yolo = Yolov3(class_names=class_names)
    yolo.create_model(anchors=anchors)

    binary_weight_list = []

    for i in range(len(train_label)):
        binary_weight_list.append(
            get_class_weight(
            train_label[i][..., 4:5],
            method='binary'
            )
        )

    binary_weight_list = [0.1]*3


    ignore_thresh = 0.7
    use_focal_loss = True

    loss_weight = {
        "xy":1,
        "wh":1,
        "conf":5,
        "prob":1
        }

    loss_fn = yolo.loss(
        binary_weight_list,
        loss_weight=loss_weight,
        ignore_thresh=ignore_thresh
        )
    metrics = yolo.metrics("obj+iou+class")

    yolo.model.compile(
        optimizer = optimizer,
        #optimizer=SGD(learning_rate=1e-10, momentum=0.9, decay=5e-4),
        loss = loss_fn,
        metrics = metrics
        )
    
    train_history = yolo.model.fit(train_img,
                                    train_label,
                                    epochs = 50,
                                    batch_size=5,
                                    verbose=1,
                                    validation_data=(val_img, val_label),
                                    callbacks=[callback]
                                    )
    return yolo
######### Build Model Yolov3 ###############



############## Yolov4 #################
class DarknetConv2D(Conv2D):
    '''Convolution2D with Darknet parameters.
    '''
    __doc__ += Conv2D.__doc__
    def __init__(self, *args, **kwargs):
        kwargs["kernel_initializer"] = RandomNormal(mean=0.0, stddev=0.02)
        if kwargs.get("strides") == (2, 2):
            kwargs["padding"] = "valid"
        else:
            kwargs["padding"] = "same"
        super().__init__(*args, **kwargs)

class Mish(Layer):
    '''
    Mish Activation Function.
    `mish(x) = x * tanh(softplus(x))`
    Examples:
        >>> input_tensor = Input(input_shape)
        >>> output = Mish()(input_tensor)
    '''

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True
        self.trainable = False

    def call(self, inputs):
        return inputs * tanh(softplus(inputs))

def mish(x):
    return x * tf.math.tanh(tf.math.softplus(x))
def upsample(input_layer):
    return tf.image.resize(input_layer, (input_layer.shape[1] * 2, input_layer.shape[2] * 2), method='nearest')
# class Yolov4(object):
#     def __init__(self,
#                  input_shape=(416, 416, 3),
#                  class_names=[]):
#         self.input_shape = input_shape
#         self.grid_shape = input_shape[0]//32, input_shape[1]//32
#         self.abox_num = 3
#         self.class_names = class_names
#         self.class_num = len(class_names)
#         self.fpn_layers = 3
#         self.anchors = None
#         self.model = None
#         self.file_names = None

#     def DarknetConv2D_BN_Leaky(tensor, *args, **kwargs):
#         '''Darknet Convolution2D followed by BatchNormalization and LeakyReLU.'''
#         bn_name = None
#         acti_name = None
#         if "name" in kwargs:
#             name = kwargs["name"]
#             kwargs["name"] = name + "_conv"
#             bn_name = name + "_bn"
#             acti_name = name + "_leaky"
#         kwargs["use_bias"] = False

#         tensor = DarknetConv2D(*args, **kwargs)(tensor)
#         tensor = BatchNormalization(name=bn_name)(tensor)
#         tensor = LeakyReLU(alpha=0.1, name=acti_name)(tensor)

#         return tensor

#     def conv2d_bn_mish(tensor, *args, **kwargs):
#         '''Darknet Convolution2D followed by BatchNormalization and Mish.
#         '''
#         bn_name = None
#         acti_name = None
#         if "name" in kwargs:
#             name = kwargs["name"]
#             kwargs["name"] = name + "_conv"
#             bn_name = name + "_bn"
#             acti_name = name + "_mish"
#         kwargs["use_bias"] = False

#         tensor = DarknetConv2D(*args, **kwargs)(tensor)
#         tensor = BatchNormalization(name=bn_name)(tensor)
#         tensor = Mish(name=acti_name)(tensor)

#         return tensor

#     '''
#     CSP Residual Block
#     '''
#     def resblock_module(self, tensor, mid_filters, out_filters, name="block1"):
#         '''CSPDarkNet53 residual block module.'''
#         skip_tensor = tensor
#         tensor = self.conv2d_bn_mish(
#             tensor, mid_filters, 1, name=name + "_1x1")
#         tensor = self.conv2d_bn_mish(
#             tensor, out_filters, 3, name=name + "_3x3")
#         tensor = Add(name=name + "_add")([tensor, skip_tensor])
#         return tensor

#     def resstage_module(self, tensor, num_filters, num_blocks,
#                     is_narrow=True, name="block1"):
#         '''CSPDarkNet53 residual stage module.'''
#         mid_filters = num_filters//2 if is_narrow else num_filters

#         tensor = ZeroPadding2D(((1, 0), (1, 0)), name=name + "_pad")(tensor)
#         tensor = self.conv2d_bn_mish(
#             tensor, num_filters, 3, strides=(2, 2), name=name + "_dn")
#         cross_tensor = self.conv2d_bn_mish(
#             tensor, mid_filters, 1, name=name + "_cross")
#         tensor = self.conv2d_bn_mish(
#             tensor, mid_filters, 1, name=name + "_pre")
#         for i_block in range(num_blocks):
#             tensor = self.resblock_module(
#                 tensor, num_filters//2, mid_filters,
#                 name=f"{name}_block{i_block + 1}")
#         tensor = self.conv2d_bn_mish(
#             tensor, mid_filters, 1, name=name + "_post")
#         tensor = Concatenate(name=name + "_concat")([tensor, cross_tensor])
#         tensor = self.conv2d_bn_mish(
#             tensor, num_filters, 1, name=name + "_out")
#         return tensor
    
#     '''
#     08. CSP Backbone darknet
#     '''
#     def CSP_Backbone_darknet(self, input_tensor):
#         '''CSPDarkNet53 model body.'''
#         x = self.conv2d_bn_mish(input_tensor, 32, 3, name="conv1")
#         x = self.resstage_module(x, 64, 1, False, name="stage1")
#         x = self.resstage_module(x, 128, 2, name="stage2")
#         x = self.resstage_module(x, 256, 8, name="stage3")
#         x = self.resstage_module(x, 512, 8, name="stage4")
#         x = self.resstage_module(x, 1024, 4, name="stage5")
#         return x


#     def make_last_layers(self, tensor, num_filters, name="last1"):
#         '''5 DarknetConv2D_BN_Leaky layers followed by a Conv2D layer'''
#         tensor = self.DarknetConv2D_BN_Leaky(
#             tensor, num_filters, 1, name=f"{name}_1")
#         tensor = self.DarknetConv2D_BN_Leaky(
#             tensor, num_filters*2, 3, name=f"{name}_2")
#         tensor = self.DarknetConv2D_BN_Leaky(
#             tensor, num_filters, 1, name=f"{name}_3")
#         tensor = self.DarknetConv2D_BN_Leaky(
#             tensor, num_filters*2, 3, name=f"{name}_4")
#         tensor = self.DarknetConv2D_BN_Leaky(
#             tensor, num_filters, 1, name=f"{name}_5")

#         return tensor
    
#     '''
#     SPP Module
#     '''
#     def spp_module(self, tensor, pool_size_list=[(13, 13), (9, 9), (5, 5)],
#                 name="spp"):
#         '''Spatial pyramid pooling module.'''
#         maxpool_tensors = []
#         for i_pool, pool_size in enumerate(pool_size_list):
#             maxpool_tensors.append(MaxPooling2D(
#                 pool_size=pool_size, strides=(1, 1),
#                 padding="same", name=f"{name}_pool{i_pool + 1}")(tensor))
#         tensor = Concatenate(name=name + "_concat")([*maxpool_tensors, tensor])
#         return tensor
    
#     '''
#     YOLO Neck
#     '''
#     def yolo_neck(self, input_shape=(608, 608, 3),
#                 pretrained_darknet=None,
#                 pretrained_weights=None):
#         '''Create YOLOv4 body in tf.keras.'''
#         input_tensor = Input(input_shape)
#         darknet = Model(input_tensor, self.CSP_Backbone_darknet(input_tensor))
#         if pretrained_darknet is not None:
#             darknet.set_weights(pretrained_darknet.get_weights())

#         tensor_s = self.DarknetConv2D_BN_Leaky(
#             darknet.output, 512, 1, name="pan_td1_1")
#         tensor_s = self.DarknetConv2D_BN_Leaky(
#             tensor_s, 1024, 3, name="pan_td1_2")
#         tensor_s = self.DarknetConv2D_BN_Leaky(
#             tensor_s, 512, 1, name="pan_td1_spp_pre")
#         tensor_s = self.spp_module(tensor_s, name="pan_td1_spp")
#         tensor_s = self.DarknetConv2D_BN_Leaky(
#             tensor_s, 512, 1, name="pan_td1_3")
#         tensor_s = self.DarknetConv2D_BN_Leaky(
#             tensor_s, 1024, 3, name="pan_td1_4")
#         tensor_s = self.DarknetConv2D_BN_Leaky(
#             tensor_s, 512, 1, name="pan_td1_5")

#         tensor_s_up = self.DarknetConv2D_BN_Leaky(
#             tensor_s, 256, 1, name="pan_td1_up")
#         tensor_s_up = UpSampling2D(2, name="pan_td1_up")(tensor_s_up)

#         tensor_m = self.DarknetConv2D_BN_Leaky(
#             darknet.layers[204].output, 256, 1, name="pan_td2_pre")
#         tensor_m = Concatenate(name="pan_td1_concat")([tensor_m, tensor_s_up])
#         tensor_m = self.make_last_layers(tensor_m, 256, name="pan_td2")

#         tensor_m_up = self.DarknetConv2D_BN_Leaky(
#             tensor_m, 128, 1, name="pan_td2_up")
#         tensor_m_up = UpSampling2D(2, name="pan_td2_up")(tensor_m_up)

#         tensor_l = self.DarknetConv2D_BN_Leaky(
#             darknet.layers[131].output, 128, 1, name="pan_td3_pre")
#         tensor_l = Concatenate(name="pan_td2_concat")([tensor_l, tensor_m_up])
#         tensor_l = self.make_last_layers(tensor_l, 128, name="pan_td3")

#         output_l = self.DarknetConv2D_BN_Leaky(
#             tensor_l, 256, 3, name="pan_out_l")

#         tensor_l_dn = ZeroPadding2D(
#             ((1, 0),(1, 0)), name="pan_bu1_dn_pad")(tensor_l)
#         tensor_l_dn = self.DarknetConv2D_BN_Leaky(
#             tensor_l_dn, 256, 3, strides=(2, 2), name="pan_bu1_dn")
#         tensor_m = Concatenate(name="pan_bu1_concat")([tensor_l_dn, tensor_m])
#         tensor_m = self.make_last_layers(tensor_m, 256, name="pan_bu1")

#         output_m = self.DarknetConv2D_BN_Leaky(
#             tensor_m, 512, 3, name="pan_out_m")

#         tensor_m_dn = ZeroPadding2D(
#             ((1, 0),(1, 0)), name="pan_bu2_dn_pad")(tensor_m)
#         tensor_m_dn = self.DarknetConv2D_BN_Leaky(
#             tensor_m_dn, 512, 3, strides=(2, 2), name="pan_bu2_dn")
#         tensor_s = Concatenate(name="pan_bu2_concat")([tensor_m_dn, tensor_s])
#         tensor_s = self.make_last_layers(tensor_s, 512, name="pan_bu2")

#         output_s = self.DarknetConv2D_BN_Leaky(
#             tensor_s, 1024, 3, name="pan_out_s")

#         model = Model(input_tensor, [output_s, output_m, output_l])

#         # if pretrained_weights is not None:
#         #     if pretrained_weights == "ms_coco":
#         #         pretrained_weights = get_file(
#         #             "tf_keras_yolov4_body.h5",
#         #             WEIGHTS_PATH_YOLOV4_BODY,
#         #             cache_subdir="models")
#         #     model.load_weights(pretrained_weights)
        
#         return model
    
#     '''
#     11. Head
#     '''
#     def yolo_head(self, model_body, class_num=10, 
#                 anchors=[[0.89663461, 0.78365384],
#                         [0.37500000, 0.47596153],
#                         [0.27884615, 0.21634615],
#                         [0.14182692, 0.28605769],
#                         [0.14903846, 0.10817307],
#                         [0.07211538, 0.14663461],
#                         [0.07932692, 0.05528846],
#                         [0.03846153, 0.07211538],
#                         [0.02403846, 0.03125000]]):
#         anchors = np.array(anchors)
#         inputs = model_body.input
#         output = model_body.output
#         tensor_num = len(output)

#         if len(anchors)%tensor_num > 0:
#             raise ValueError(("The total number of anchor boxs"
#                             " should be a multiple of the number(%s)"
#                             " of output tensors") % tensor_num)    
#         abox_num = len(anchors)//tensor_num

#         outputs_list = []
#         for tensor_i, output_tensor in enumerate(output):
#             output_list = []
#             start_i = tensor_i*abox_num
#             for box in anchors[start_i:start_i + abox_num]:
#                 xy_output = DarknetConv2D(2, 1,
#                                 activation='sigmoid')(output_tensor)
#                 wh_output = DarknetConv2D(2, 1,
#                                 activation='exponential')(output_tensor)
#                 wh_output = wh_output * box
#                 c_output = DarknetConv2D(1, 1,
#                                 activation='sigmoid')(output_tensor)
#                 p_output = DarknetConv2D(class_num, 1,
#                                 activation='sigmoid')(output_tensor)
#                 output_list += [xy_output,
#                                 wh_output,
#                                 c_output,
#                                 p_output]

#             outputs = concatenate(output_list, axis=-1)
#             outputs_list.append(outputs)
        
#         model = Model(inputs, outputs_list)    

#         return model
    
#     def cal_iou(self, xywh_true, xywh_pred, grid_shape, return_ciou=False):
#         '''Calculate IOU of two tensors.
#         return shape: (N, S, S, B)[, (N, S, S, B)]
#         '''
#         grid_shape = np.array(grid_shape[::-1])
#         xy_true = xywh_true[..., 0:2]/grid_shape # N*S*S*1*2
#         wh_true = xywh_true[..., 2:4]

#         xy_pred = xywh_pred[..., 0:2]/grid_shape # N*S*S*B*2
#         wh_pred = xywh_pred[..., 2:4]
        
#         half_xy_true = wh_true / 2.
#         mins_true    = xy_true - half_xy_true
#         maxes_true   = xy_true + half_xy_true

#         half_xy_pred = wh_pred / 2.
#         mins_pred    = xy_pred - half_xy_pred
#         maxes_pred   = xy_pred + half_xy_pred       
        
#         intersect_mins  = tf.maximum(mins_pred,  mins_true)
#         intersect_maxes = tf.minimum(maxes_pred, maxes_true)
#         intersect_wh    = tf.maximum(intersect_maxes - intersect_mins, 0.)
#         intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]
        
#         true_areas = wh_true[..., 0] * wh_true[..., 1]
#         pred_areas = wh_pred[..., 0] * wh_pred[..., 1]

#         union_areas = pred_areas + true_areas - intersect_areas
#         iou_scores  = intersect_areas/(union_areas + epsilon)

#         if return_ciou:
#             enclose_mins = tf.minimum(mins_pred,  mins_true)
#             enclose_maxes = tf.maximum(maxes_pred, maxes_true)

#             enclose_wh = enclose_maxes - enclose_mins
#             enclose_c2 = (tf.pow(enclose_wh[..., 0], 2)
#                         + tf.pow(enclose_wh[..., 1], 2))

#             p_rho2 = (tf.pow(xy_true[..., 0] - xy_pred[..., 0], 2)
#                     + tf.pow(xy_true[..., 1] - xy_pred[..., 1], 2))

#             atan_true = tf.atan(wh_true[..., 0] / (wh_true[..., 1] + epsilon))
#             atan_pred = tf.atan(wh_pred[..., 0] / (wh_pred[..., 1] + epsilon))

#             v_nu = 4.0 / (math.pi ** 2) * tf.pow(atan_true - atan_pred, 2)
#             a_alpha = v_nu / (1 - iou_scores + v_nu)

#             ciou_scores = iou_scores - p_rho2/enclose_c2 - a_alpha*v_nu

#             return iou_scores, ciou_scores

#         return iou_scores
    

#     '''
#     13. Yolo Loss Function
#     '''
#     def wrap_yolo_loss(self,grid_shape,
#                     bbox_num,
#                     class_num,
#                     anchors=None,
#                     binary_weight=1,
#                     loss_weight=[1, 1, 1],
#                     wh_reg_weight=0.01,
#                     ignore_thresh=.6,
#                     truth_thresh=1,
#                     label_smooth=0,
#                     focal_loss_gamma=2):
#         '''Wrapped YOLOv4 loss function.'''
#         def yolo_loss(y_true, y_pred):
#             if anchors is None:
#                 panchors = 1
#             else:
#                 panchors = tf.reshape(anchors, (1, 1, 1, bbox_num, 2))

#             y_true = tf.reshape(
#                 y_true,
#                 (-1, *grid_shape, 1, 5 + class_num)) # N*S*S*1*(5+C)
#             y_pred = tf.reshape(
#                 y_pred,
#                 (-1, *grid_shape, bbox_num, 5 + class_num)) # N*S*S*B*(5+C)

#             xywh_true = y_true[..., :4] # N*S*S*1*4
#             xywh_pred = y_pred[..., :4] # N*S*S*B*4

#             iou_scores, ciou_scores = self.cal_iou(
#                 xywh_true, xywh_pred, grid_shape, return_ciou=True) # N*S*S*B

#             response_mask = tf.one_hot(tf.argmax(iou_scores, axis=-1),
#                                     depth=bbox_num,
#                                     dtype=xywh_true.dtype) # N*S*S*B

#             has_obj_mask = y_true[..., 4]*response_mask # N*S*S*B

#             if truth_thresh < 1:
#                 truth_mask = tf.cast(
#                     iou_scores > truth_thresh,
#                     iou_scores.dtype) # N*S*S*B
#                 has_obj_mask = has_obj_mask + truth_mask*(1 - has_obj_mask)
#             has_obj_mask_exp = tf.expand_dims(has_obj_mask, axis=-1) # N*S*S*B*1

#             no_obj_mask = tf.cast(
#                 iou_scores < ignore_thresh,
#                 iou_scores.dtype) # N*S*S*B
#             no_obj_mask = (1 - has_obj_mask)*no_obj_mask # N*S*S*B

#             box_loss = tf.reduce_sum(
#                 tf.reduce_mean(
#                 has_obj_mask # N*S*S*B
#                 *(1 - ciou_scores), # N*S*S*B
#                 axis=0))

#             c_pred = y_pred[..., 4] # N*S*S*B
#             c_pred = tf.clip_by_value(c_pred, epsilon, 1 - epsilon)

#             if label_smooth > 0:
#                 label = 1 - label_smooth

#                 has_obj_c_loss = -tf.reduce_sum(
#                     tf.reduce_mean(
#                     has_obj_mask # N*S*S*B
#                     *(tf.math.abs(label - c_pred)**focal_loss_gamma)
#                     *tf.math.log(1 - tf.math.abs(label - c_pred)),
#                     axis=0))
                
#                 no_obj_c_loss = -tf.reduce_sum(
#                     tf.reduce_mean(
#                     no_obj_mask # N*S*S*B
#                     *(tf.math.abs(label_smooth - c_pred)**focal_loss_gamma)
#                     *tf.math.log(1 - tf.math.abs(label_smooth - c_pred)),
#                     axis=0))
#             else:
#                 has_obj_c_loss = -tf.reduce_sum(
#                     tf.reduce_mean(
#                     has_obj_mask # N*S*S*B
#                     *((1 - c_pred)**focal_loss_gamma)
#                     *tf.math.log(c_pred),
#                     axis=0))

#                 no_obj_c_loss = -tf.reduce_sum(
#                     tf.reduce_mean(
#                     no_obj_mask # N*S*S*B
#                     *((c_pred)**focal_loss_gamma)
#                     *tf.math.log(1 - c_pred),
#                     axis=0))
            
#             c_loss = has_obj_c_loss + binary_weight*no_obj_c_loss

#             p_true = y_true[..., -class_num:] # N*S*S*1*C
#             p_pred = y_pred[..., -class_num:] # N*S*S*B*C
#             p_pred = tf.clip_by_value(p_pred, epsilon, 1 - epsilon)
#             p_loss = -tf.reduce_sum(
#                 tf.reduce_mean(
#                     has_obj_mask_exp # N*S*S*B*1
#                     *(p_true*tf.math.log(p_pred)
#                     + (1 - p_true)*tf.math.log(1 - p_pred)), # N*S*S*B*C
#                     axis=0))
            
#             wh_pred = y_pred[..., 2:4]/panchors # N*S*S*B*2
#             wh_pred = tf.math.log(wh_pred) # N*S*S*B*2

#             wh_reg = tf.reduce_sum(
#                 tf.reduce_mean(wh_pred**2, axis=0))

#             loss = (loss_weight[0]*box_loss
#                     + loss_weight[1]*c_loss
#                     + loss_weight[2]*p_loss
#                     + wh_reg_weight*wh_reg)

#             return loss

#         return yolo_loss
    

#     def wrap_obj_acc(self, grid_shape, bbox_num, class_num):
#         def obj_acc(self, y_true, y_pred):
#             y_true = tf.reshape(
#                 y_true,
#                 (-1, *grid_shape, 1, 5 + class_num)) # N*S*S*1*5+C
#             y_pred = tf.reshape(
#                 y_pred,
#                 (-1, *grid_shape, bbox_num, 5 + class_num)) # N*S*S*B*5+C
            
#             c_true = y_true[..., 4] # N*S*S*1
#             c_pred = tf.reduce_max(y_pred[..., 4], # N*S*S*B
#                                 axis=-1,
#                                 keepdims=True) # N*S*S*1

#             bi_acc = binary_accuracy(c_true, c_pred)

#             return bi_acc
#         return obj_acc


#     def wrap_mean_iou(self, grid_shape, bbox_num, class_num):
#         def mean_iou(self, y_true, y_pred):
#             y_true = tf.reshape(
#                 y_true,
#                 (-1, *grid_shape, 1, 5 + class_num)) # N*S*S*1*5+C
#             y_pred = tf.reshape(
#                 y_pred,
#                 (-1, *grid_shape, bbox_num, 5 + class_num)) # N*S*S*B*5+C

#             has_obj_mask = y_true[..., 4] # N*S*S*1
            
#             xywh_true = y_true[..., :4] # N*S*S*1*4
#             xywh_pred = y_pred[..., :4] # N*S*S*B*4

#             iou_scores = self.cal_iou(xywh_true, xywh_pred, grid_shape) # N*S*S*B
#             iou_scores = tf.reduce_max(iou_scores, axis=-1, keepdims=True) # N*S*S*1
#             iou_scores = iou_scores*has_obj_mask # N*S*S*1

#             num_p = tf.reduce_sum(has_obj_mask)

#             return tf.reduce_sum(iou_scores)/(num_p + epsilon)
#         return mean_iou

#     def wrap_class_acc(self, grid_shape, bbox_num, class_num):
#         def class_acc(self, y_true, y_pred):
#             y_true = tf.reshape(
#                 y_true,
#                 (-1, *grid_shape, 1, 5 + class_num)) # N*S*S*1*5+C
#             y_pred = tf.reshape(
#                 y_pred,
#                 (-1, *grid_shape, bbox_num, 5 + class_num)) # N*S*S*B*5+C

#             has_obj_mask = y_true[..., 4] # N*S*S*1

#             pi_true = tf.argmax(y_true[..., -class_num:], # N*S*S*1*C
#                                 axis=-1) # N*S*S*1
#             pi_pred = tf.argmax(y_pred[..., -class_num:], # N*S*S*B*C
#                                 axis=-1) # N*S*S*B
            
#             equal_mask = tf.cast(pi_true == pi_pred,
#                                 dtype=y_true.dtype) # N*S*S*B
#             equal_mask = equal_mask*has_obj_mask # N*S*S*B

#             num_p = tf.reduce_sum(has_obj_mask)*bbox_num

#             return tf.reduce_sum(equal_mask)/(num_p + epsilon)
#         return class_acc
    

#     '''
#     Model Create
#     '''
#     def create_model(self,
#                     anchors=[[0.89663461, 0.78365384],
#                             [0.37500000, 0.47596153],
#                             [0.27884615, 0.21634615],
#                             [0.14182692, 0.28605769],
#                             [0.14903846, 0.10817307],
#                             [0.07211538, 0.14663461],
#                             [0.07932692, 0.05528846],
#                             [0.03846153, 0.07211538],
#                             [0.02403846, 0.03125000]],
#                     backbone="full_darknet",
#                     pretrained_weights=None,
#                     pretrained_darknet="ms_coco"):
        
#         if isinstance(pretrained_darknet, str):
#             pre_body_weights = pretrained_darknet
#             pretrained_darknet = None
#         else:
#             pre_body_weights = None
        
#         model_body = self.yolo_neck(self.input_shape,
#             pretrained_weights=pre_body_weights)

#         if pretrained_darknet is not None:
#             model_body.set_weights(pretrained_darknet.get_weights())
#         self.model = self.yolo_head(model_body,
#                             self.class_num,
#                             anchors)
        
#         if pretrained_weights is not None:
#             self.model.load_weights(pretrained_weights)
#         self.anchors = anchors
#         self.grid_shape = self.model.output[0].shape[1:3]
#         self.fpn_layers = len(self.model.output)
#         self.abox_num = len(self.anchors)//self.fpn_layers

#     '''
#     Loss Create
#     '''
#     def loss(self,
#             binary_weight=1,
#             loss_weight=[1, 1, 5, 1],
#             ignore_thresh=0.6):

#         if (not isinstance(binary_weight, Iterable)
#             or len(binary_weight) != self.fpn_layers):
#             binary_weight = [binary_weight]*self.fpn_layers
        
#         if isinstance(loss_weight, dict):
#             loss_weight_list = []
#             loss_weight_list.append(loss_weight["xy"])
#             loss_weight_list.append(loss_weight["wh"])
#             loss_weight_list.append(loss_weight["conf"])
#             loss_weight_list.append(loss_weight["prob"])
#             loss_weight = loss_weight_list
        
#         loss_list = []
#         for fpn_id in range(self.fpn_layers):
#             grid_amp = 2**(fpn_id)
#             grid_shape = (self.grid_shape[0]*grid_amp,
#                         self.grid_shape[1]*grid_amp)
#             anchors_id = self.abox_num*fpn_id
#             loss_list.append(self.wrap_yolo_loss(
#                 grid_shape=grid_shape,
#                 bbox_num=self.abox_num, 
#                 class_num=self.class_num,
#                 anchors=self.anchors[
#                     anchors_id:anchors_id + self.abox_num],
#                 binary_weight=binary_weight[fpn_id],
#                 loss_weight=loss_weight,
#                 ignore_thresh=ignore_thresh))
#         return loss_list
    
#     '''
#     Metrics Create
#     '''
#     def metrics(self, type="obj_acc"):
        
        
#         metrics_list = [[] for _ in range(self.fpn_layers)]
#         for fpn_id in range(self.fpn_layers):
#             grid_amp = 2**(fpn_id)
#             grid_shape = (self.grid_shape[0]*grid_amp,
#                             self.grid_shape[1]*grid_amp)
            
#             if "obj" in type:
#                 metrics_list[fpn_id].append(
#                     self.wrap_obj_acc(
#                         grid_shape,
#                         self.abox_num, 
#                         self.class_num))
#             if "iou" in type:
#                 metrics_list[fpn_id].append(
#                     self.wrap_mean_iou(
#                         grid_shape,
#                         self.abox_num, 
#                         self.class_num))
#             if "class" in type:
#                 metrics_list[fpn_id].append(
#                     self.wrap_class_acc(
#                         grid_shape,
#                         self.abox_num, 
#                         self.class_num))
        
#         return metrics_list


    
class Yolov4(object):

    def __init__(self,
                 input_shape=(416, 416, 3),
                 class_names=[]):
        self.input_shape = input_shape
        self.grid_shape = input_shape[0]//32, input_shape[1]//32
        self.abox_num = 3
        self.class_names = class_names
        self.class_num = len(class_names)
        self.fpn_layers = 3
        self.anchors = None
        self.model = None
        self.file_names = None


    '''
    CSP Residual Block
    If train model on GPU then activate_type = mish, CPU: activate_type = leaky 
    '''
    def DarknetConv2D_BN_Leaky(self, input_layer, filters_shape, downsample=False, activate=True, bn=True, activate_type='leaky'):
        if downsample:
            input_layer = ZeroPadding2D(((1, 0), (1, 0)))(input_layer)
            padding = 'valid'
            strides = 2
        else:
            strides = 1
            padding = 'same'

        conv = Conv2D(filters=filters_shape[-1], kernel_size = filters_shape[0], strides=strides,
                    padding=padding, use_bias=not bn, kernel_regularizer=l2(0.0005),
                    kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                    bias_initializer=tf.constant_initializer(0.))(input_layer)
        if bn:
            conv = BatchNormalization()(conv)
        if activate == True:
            if activate_type == "leaky":
                conv = LeakyReLU(alpha=0.1)(conv)
            elif activate_type == "mish":
                conv = mish(conv)

        return conv
    
    '''
    DarknetResidual
    '''
    def DarknetResidual(self, input_layer, input_channel, filter_num1, filter_num2, activate_type='leaky'):
        short_cut = input_layer
        conv = self.DarknetConv2D_BN_Leaky(input_layer, filters_shape=(1, 1, input_channel, filter_num1), activate_type=activate_type)
        conv = self.DarknetConv2D_BN_Leaky(conv       , filters_shape=(3, 3, filter_num1,   filter_num2), activate_type=activate_type)

        residual_output = short_cut + conv
        return residual_output


    '''
    CSP Backbone darknet
    Note: if use GPU then activate_type = mish, CPU: activate_type = leaky
    '''
    def CSP_Backbone_darknet(self, input_tensor):
        input_tensor = self.DarknetConv2D_BN_Leaky(input_tensor, (3, 3,  3,  32), activate_type="mish")
        input_tensor = self.DarknetConv2D_BN_Leaky(input_tensor, (3, 3, 32,  64), downsample=True, activate_type="mish")

        route = input_tensor
        route = self.DarknetConv2D_BN_Leaky(route, (1, 1, 64, 64), activate_type="mish")
        input_tensor = self.DarknetConv2D_BN_Leaky(input_tensor, (1, 1, 64, 64), activate_type="mish")
        for i in range(1):
            input_tensor = self.DarknetResidual(input_tensor,  64,  32, 64, activate_type="mish")
        input_tensor = self.DarknetConv2D_BN_Leaky(input_tensor, (1, 1, 64, 64), activate_type="mish")

        input_tensor = tf.concat([input_tensor, route], axis=-1)
        input_tensor = self.DarknetConv2D_BN_Leaky(input_tensor, (1, 1, 128, 64), activate_type="mish")
        input_tensor = self.DarknetConv2D_BN_Leaky(input_tensor, (3, 3, 64, 128), downsample=True, activate_type="mish")
        route = input_tensor
        route = self.DarknetConv2D_BN_Leaky(route, (1, 1, 128, 64), activate_type="mish")
        input_tensor = self.DarknetConv2D_BN_Leaky(input_tensor, (1, 1, 128, 64), activate_type="mish")
        for i in range(2):
            input_tensor = self.DarknetResidual(input_tensor, 64,  64, 64, activate_type="mish")
        input_tensor = self.DarknetConv2D_BN_Leaky(input_tensor, (1, 1, 64, 64), activate_type="mish")
        input_tensor = tf.concat([input_tensor, route], axis=-1)

        input_tensor = self.DarknetConv2D_BN_Leaky(input_tensor, (1, 1, 128, 128), activate_type="mish")
        input_tensor = self.DarknetConv2D_BN_Leaky(input_tensor, (3, 3, 128, 256), downsample=True, activate_type="mish")
        route = input_tensor
        route = self.DarknetConv2D_BN_Leaky(route, (1, 1, 256, 128), activate_type="mish")
        input_tensor = self.DarknetConv2D_BN_Leaky(input_tensor, (1, 1, 256, 128), activate_type="mish")
        for i in range(8):
            input_tensor = self.DarknetResidual(input_tensor, 128, 128, 128, activate_type="mish")
        input_tensor = self.DarknetConv2D_BN_Leaky(input_tensor, (1, 1, 128, 128), activate_type="mish")
        input_tensor = tf.concat([input_tensor, route], axis=-1)

        input_tensor = self.DarknetConv2D_BN_Leaky(input_tensor, (1, 1, 256, 256), activate_type="mish")
        route_1 = input_tensor
        input_tensor = self.DarknetConv2D_BN_Leaky(input_tensor, (3, 3, 256, 512), downsample=True, activate_type="mish")
        route = input_tensor
        route = self.DarknetConv2D_BN_Leaky(route, (1, 1, 512, 256), activate_type="mish")
        input_tensor = self.DarknetConv2D_BN_Leaky(input_tensor, (1, 1, 512, 256), activate_type="mish")
        for i in range(8):
            input_tensor = self.DarknetResidual(input_tensor, 256, 256, 256, activate_type="mish")
        input_tensor = self.DarknetConv2D_BN_Leaky(input_tensor, (1, 1, 256, 256), activate_type="mish")
        input_tensor = tf.concat([input_tensor, route], axis=-1)

        input_tensor = self.DarknetConv2D_BN_Leaky(input_tensor, (1, 1, 512, 512), activate_type="mish")
        route_2 = input_tensor
        input_tensor = self.DarknetConv2D_BN_Leaky(input_tensor, (3, 3, 512, 1024), downsample=True, activate_type="mish")
        route = input_tensor
        route = self.DarknetConv2D_BN_Leaky(route, (1, 1, 1024, 512), activate_type="mish")
        input_tensor = self.DarknetConv2D_BN_Leaky(input_tensor, (1, 1, 1024, 512), activate_type="mish")
        for i in range(4):
            input_tensor = self.DarknetResidual(input_tensor, 512, 512, 512, activate_type="mish")
        input_tensor = self.DarknetConv2D_BN_Leaky(input_tensor, (1, 1, 512, 512), activate_type="mish")
        input_tensor = tf.concat([input_tensor, route], axis=-1)

        input_tensor = self.DarknetConv2D_BN_Leaky(input_tensor, (1, 1, 1024, 1024), activate_type="mish")
        input_tensor = self.DarknetConv2D_BN_Leaky(input_tensor, (1, 1, 1024, 512))
        input_tensor = self.DarknetConv2D_BN_Leaky(input_tensor, (3, 3, 512, 1024))
        input_tensor = self.DarknetConv2D_BN_Leaky(input_tensor, (1, 1, 1024, 512))

        input_tensor = tf.concat([tf.nn.max_pool(input_tensor, ksize=13, padding='SAME', strides=1), tf.nn.max_pool(input_tensor, ksize=9, padding='SAME', strides=1)
                                , tf.nn.max_pool(input_tensor, ksize=5, padding='SAME', strides=1), input_tensor], axis=-1)
        input_tensor = self.DarknetConv2D_BN_Leaky(input_tensor, (1, 1, 2048, 512))
        input_tensor = self.DarknetConv2D_BN_Leaky(input_tensor, (3, 3, 512, 1024))
        input_tensor = self.DarknetConv2D_BN_Leaky(input_tensor, (1, 1, 1024, 512))

        return route_1, route_2, input_tensor
    
    '''
    YOLO Neck
    '''
    def yolo_neck(self, input_shape=(608, 608, 3),
              pretrained_darknet=None,
              pretrained_weights=None):
        '''Create YOLOv4 body in tf.keras.'''
        input_tensor = Input(input_shape)
        # darknet = Model(input_tensor, CSP_Backbone_darknet(input_tensor))
        route_1, route_2, conv = self.CSP_Backbone_darknet(input_tensor)
        
        # if pretrained_darknet is not None:
        #     darknet.set_weights(pretrained_darknet.get_weights())
        
        tensor_s_up = self.DarknetConv2D_BN_Leaky(conv, (1, 1, 512, 256))
        tensor_s_up = upsample(tensor_s_up)
        
        tensor_m = self.DarknetConv2D_BN_Leaky(route_2, (1, 1, 512, 256))
        tensor_m = tf.concat([tensor_m, tensor_s_up], axis=-1)
        
        tensor_m = self.DarknetConv2D_BN_Leaky(tensor_m, (1, 1, 512, 256))
        tensor_m = self.DarknetConv2D_BN_Leaky(tensor_m, (3, 3, 256, 512))
        tensor_m = self.DarknetConv2D_BN_Leaky(tensor_m, (1, 1, 512, 256))
        tensor_m = self.DarknetConv2D_BN_Leaky(tensor_m, (3, 3, 256, 512))
        tensor_m = self.DarknetConv2D_BN_Leaky(tensor_m, (1, 1, 512, 256))
        
        tensor_m_up = self.DarknetConv2D_BN_Leaky(tensor_m, (1, 1, 256, 128))
        tensor_m_up = upsample(tensor_m_up)
        
        tensor_l = self.DarknetConv2D_BN_Leaky(route_1, (1, 1, 256, 128))
        tensor_l = tf.concat([tensor_l, tensor_m_up], axis=-1)
        
        tensor_l = self.DarknetConv2D_BN_Leaky(tensor_l, (1, 1, 256, 128))
        tensor_l = self.DarknetConv2D_BN_Leaky(tensor_l, (3, 3, 128, 256))
        tensor_l = self.DarknetConv2D_BN_Leaky(tensor_l, (1, 1, 256, 128))
        tensor_l = self.DarknetConv2D_BN_Leaky(tensor_l, (3, 3, 128, 256))
        tensor_l = self.DarknetConv2D_BN_Leaky(tensor_l, (1, 1, 256, 128))
        
        output_l = self.DarknetConv2D_BN_Leaky(tensor_l, (3, 3, 128, 256))

        tensor_l_dn = ZeroPadding2D(((1, 0),(1, 0)), name="pan_bu1_dn_pad")(tensor_l)
        tensor_l_dn = self.DarknetConv2D_BN_Leaky(tensor_l_dn, (3, 3, 128, 256), downsample=True)
        
        tensor_m = tf.concat([tensor_l_dn, tensor_m], axis=-1)
        
        tensor_m = self.DarknetConv2D_BN_Leaky(tensor_m, (1, 1, 512, 256))
        tensor_m = self.DarknetConv2D_BN_Leaky(tensor_m, (3, 3, 256, 512))
        tensor_m = self.DarknetConv2D_BN_Leaky(tensor_m, (1, 1, 512, 256))
        tensor_m = self.DarknetConv2D_BN_Leaky(tensor_m, (3, 3, 256, 512))
        tensor_m = self.DarknetConv2D_BN_Leaky(tensor_m, (1, 1, 512, 256))
        
        output_m = self.DarknetConv2D_BN_Leaky(tensor_m, (3, 3, 256, 512))

        tensor_m_dn = ZeroPadding2D(((1, 0),(1, 0)), name="pan_bu2_dn_pad")(tensor_m)
        tensor_m_dn = self.DarknetConv2D_BN_Leaky(tensor_m_dn, (3, 3, 256, 512), downsample=True)
        
        tensor_s = tf.concat([tensor_m_dn, conv], axis=-1)
        
        tensor_s = self.DarknetConv2D_BN_Leaky(tensor_s, (1, 1, 1024, 512))
        tensor_s = self.DarknetConv2D_BN_Leaky(tensor_s, (3, 3, 512, 1024))
        tensor_s = self.DarknetConv2D_BN_Leaky(tensor_s, (1, 1, 1024, 512))
        tensor_s = self.DarknetConv2D_BN_Leaky(tensor_s, (3, 3, 512, 1024))
        tensor_s = self.DarknetConv2D_BN_Leaky(tensor_s, (1, 1, 1024, 512))
        
        output_s = self.DarknetConv2D_BN_Leaky(tensor_s, (3, 3, 512, 1024))

        model = Model(input_tensor, [output_s, output_m, output_l])
        
        return model
    
    '''
    Yolo Head
    '''
    def yolo_head(self, model_body, class_num=10, 
              anchors=[[0.89663461, 0.78365384],
                       [0.37500000, 0.47596153],
                       [0.27884615, 0.21634615],
                       [0.14182692, 0.28605769],
                       [0.14903846, 0.10817307],
                       [0.07211538, 0.14663461],
                       [0.07932692, 0.05528846],
                       [0.03846153, 0.07211538],
                       [0.02403846, 0.03125000]]):
        anchors = np.array(anchors)
        inputs = model_body.input
        output = model_body.output
        tensor_num = len(output)

        if len(anchors)%tensor_num > 0:
            raise ValueError(("The total number of anchor boxs"
                            " should be a multiple of the number(%s)"
                            " of output tensors") % tensor_num)    
        abox_num = len(anchors)//tensor_num

        outputs_list = []
        for tensor_i, output_tensor in enumerate(output):
            output_list = []
            start_i = tensor_i*abox_num
            for box in anchors[start_i:start_i + abox_num]:
                xy_output = DarknetConv2D(2, 1,
                                activation='sigmoid')(output_tensor)
                wh_output = DarknetConv2D(2, 1,
                                activation='exponential')(output_tensor)
                wh_output = wh_output * box
                c_output = DarknetConv2D(1, 1,
                                activation='sigmoid')(output_tensor)
                p_output = DarknetConv2D(class_num, 1,
                                activation='sigmoid')(output_tensor)
                                # activation='softmax')(output_tensor)
                output_list += [xy_output,
                                wh_output,
                                c_output,
                                p_output]

            outputs = concatenate(output_list, axis=-1)
            outputs_list.append(outputs)
        
        model = Model(inputs, outputs_list)    

        return model
    
    '''Iou'''
    def cal_iou(self, xywh_true, xywh_pred, grid_shape, return_ciou=False):
        '''Calculate IOU of two tensors.
        return shape: (N, S, S, B)[, (N, S, S, B)]
        '''
        grid_shape = np.array(grid_shape[::-1])
        xy_true = xywh_true[..., 0:2]/grid_shape # N*S*S*1*2
        wh_true = xywh_true[..., 2:4]

        xy_pred = xywh_pred[..., 0:2]/grid_shape # N*S*S*B*2
        wh_pred = xywh_pred[..., 2:4]
        
        half_xy_true = wh_true / 2.
        mins_true    = xy_true - half_xy_true
        maxes_true   = xy_true + half_xy_true

        half_xy_pred = wh_pred / 2.
        mins_pred    = xy_pred - half_xy_pred
        maxes_pred   = xy_pred + half_xy_pred       
        
        intersect_mins  = tf.maximum(mins_pred,  mins_true)
        intersect_maxes = tf.minimum(maxes_pred, maxes_true)
        intersect_wh    = tf.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]
        
        true_areas = wh_true[..., 0] * wh_true[..., 1]
        pred_areas = wh_pred[..., 0] * wh_pred[..., 1]

        union_areas = pred_areas + true_areas - intersect_areas
        iou_scores  = intersect_areas/(union_areas + epsilon)

        if return_ciou:
            enclose_mins = tf.minimum(mins_pred,  mins_true)
            enclose_maxes = tf.maximum(maxes_pred, maxes_true)

            enclose_wh = enclose_maxes - enclose_mins
            enclose_c2 = (tf.pow(enclose_wh[..., 0], 2)
                        + tf.pow(enclose_wh[..., 1], 2))

            p_rho2 = (tf.pow(xy_true[..., 0] - xy_pred[..., 0], 2)
                    + tf.pow(xy_true[..., 1] - xy_pred[..., 1], 2))

            atan_true = tf.atan(wh_true[..., 0] / (wh_true[..., 1] + epsilon))
            atan_pred = tf.atan(wh_pred[..., 0] / (wh_pred[..., 1] + epsilon))

            v_nu = 4.0 / (math.pi ** 2) * tf.pow(atan_true - atan_pred, 2)
            a_alpha = v_nu / (1 - iou_scores + v_nu)

            ciou_scores = iou_scores - p_rho2/enclose_c2 - a_alpha*v_nu

            return iou_scores, ciou_scores

        return iou_scores
    
    '''
    Yolo Loss Function
    '''
    def wrap_yolo_loss(self, grid_shape,
                   bbox_num,
                   class_num,
                   anchors=None,
                   binary_weight=1,
                   loss_weight=[1, 1, 1],
                   wh_reg_weight=0.01,
                   ignore_thresh=.6,
                   truth_thresh=1,
                   label_smooth=0,
                   focal_loss_gamma=2):
        '''Wrapped YOLOv4 loss function.'''
        def yolo_loss(y_true, y_pred):
            if anchors is None:
                panchors = 1
            else:
                panchors = tf.reshape(anchors, (1, 1, 1, bbox_num, 2))

            y_true = tf.reshape(
                y_true,
                (-1, *grid_shape, 1, 5 + class_num)) # N*S*S*1*(5+C)
            y_pred = tf.reshape(
                y_pred,
                (-1, *grid_shape, bbox_num, 5 + class_num)) # N*S*S*B*(5+C)

            xywh_true = y_true[..., :4] # N*S*S*1*4
            xywh_pred = y_pred[..., :4] # N*S*S*B*4

            iou_scores, ciou_scores = self.cal_iou(
                xywh_true, xywh_pred, grid_shape, return_ciou=True) # N*S*S*B
            
            response_mask = tf.one_hot(tf.argmax(iou_scores, axis=-1),
                                    depth=bbox_num,
                                    dtype=xywh_true.dtype) # N*S*S*B

            has_obj_mask = y_true[..., 4]*response_mask # N*S*S*B

            if truth_thresh < 1:
                truth_mask = tf.cast(
                    iou_scores > truth_thresh,
                    iou_scores.dtype) # N*S*S*B
                has_obj_mask = has_obj_mask + truth_mask*(1 - has_obj_mask)
            has_obj_mask_exp = tf.expand_dims(has_obj_mask, axis=-1) # N*S*S*B*1

            no_obj_mask = tf.cast(
                iou_scores < ignore_thresh,
                iou_scores.dtype) # N*S*S*B
            no_obj_mask = (1 - has_obj_mask)*no_obj_mask # N*S*S*B

            box_loss = tf.reduce_sum(
                tf.reduce_mean(
                has_obj_mask # N*S*S*B
                *(1 - ciou_scores), # N*S*S*B
                axis=0))

            c_pred = y_pred[..., 4] # N*S*S*B
            c_pred = tf.clip_by_value(c_pred, epsilon, 1 - epsilon)

            if label_smooth > 0:
                label = 1 - label_smooth

                has_obj_c_loss = -tf.reduce_sum(
                    tf.reduce_mean(
                    has_obj_mask # N*S*S*B
                    *(tf.math.abs(label - c_pred)**focal_loss_gamma)
                    *tf.math.log(1 - tf.math.abs(label - c_pred)),
                    axis=0))
                
                no_obj_c_loss = -tf.reduce_sum(
                    tf.reduce_mean(
                    no_obj_mask # N*S*S*B
                    *(tf.math.abs(label_smooth - c_pred)**focal_loss_gamma)
                    *tf.math.log(1 - tf.math.abs(label_smooth - c_pred)),
                    axis=0))
            else:
                has_obj_c_loss = -tf.reduce_sum(
                    tf.reduce_mean(
                    has_obj_mask # N*S*S*B
                    *((1 - c_pred)**focal_loss_gamma)
                    *tf.math.log(c_pred),
                    axis=0))

                no_obj_c_loss = -tf.reduce_sum(
                    tf.reduce_mean(
                    no_obj_mask # N*S*S*B
                    *((c_pred)**focal_loss_gamma)
                    *tf.math.log(1 - c_pred),
                    axis=0))
            
            c_loss = has_obj_c_loss + binary_weight*no_obj_c_loss

            p_true = y_true[..., -class_num:] # N*S*S*1*C
            p_pred = y_pred[..., -class_num:] # N*S*S*B*C
            p_pred = tf.clip_by_value(p_pred, epsilon, 1 - epsilon)
            p_loss = -tf.reduce_sum(
                tf.reduce_mean(
                    has_obj_mask_exp # N*S*S*B*1
                    *(p_true*tf.math.log(p_pred)
                    + (1 - p_true)*tf.math.log(1 - p_pred)), # N*S*S*B*C
                    axis=0))
            
            wh_pred = y_pred[..., 2:4]/panchors # N*S*S*B*2
            wh_pred = tf.math.log(wh_pred) # N*S*S*B*2

            wh_reg = tf.reduce_sum(
                tf.reduce_mean(wh_pred**2, axis=0))

            loss = (loss_weight[0]*box_loss
                    + loss_weight[1]*c_loss
                    + loss_weight[2]*p_loss
                    + wh_reg_weight*wh_reg)

            return loss

        return yolo_loss

    '''
    Object Accuracy
    '''
    def wrap_obj_acc(self, grid_shape, bbox_num, class_num):
        def obj_acc(y_true, y_pred):
            y_true = tf.reshape(
                y_true,
                (-1, *grid_shape, 1, 5 + class_num)) # N*S*S*1*5+C
            y_pred = tf.reshape(
                y_pred,
                (-1, *grid_shape, bbox_num, 5 + class_num)) # N*S*S*B*5+C
            
            c_true = y_true[..., 4] # N*S*S*1
            c_pred = tf.reduce_max(y_pred[..., 4], # N*S*S*B
                                axis=-1,
                                keepdims=True) # N*S*S*1

            bi_acc = binary_accuracy(c_true, c_pred)

            return bi_acc
        return obj_acc
    
    '''
    Mean IoU
    '''
    def wrap_mean_iou(self, grid_shape, bbox_num, class_num):
        def mean_iou(self, y_true, y_pred):
            y_true = tf.reshape(
                y_true,
                (-1, *grid_shape, 1, 5 + class_num)) # N*S*S*1*5+C
            y_pred = tf.reshape(
                y_pred,
                (-1, *grid_shape, bbox_num, 5 + class_num)) # N*S*S*B*5+C

            has_obj_mask = y_true[..., 4] # N*S*S*1
            
            xywh_true = y_true[..., :4] # N*S*S*1*4
            xywh_pred = y_pred[..., :4] # N*S*S*B*4

            iou_scores = self.cal_iou(xywh_true, xywh_pred, grid_shape) # N*S*S*B
            iou_scores = tf.reduce_max(iou_scores, axis=-1, keepdims=True) # N*S*S*1
            iou_scores = iou_scores*has_obj_mask # N*S*S*1

            num_p = tf.reduce_sum(has_obj_mask)

            return tf.reduce_sum(iou_scores)/(num_p + epsilon)
        return mean_iou


    def wrap_class_acc(self, grid_shape, bbox_num, class_num):
        def class_acc(y_true, y_pred):
            y_true = tf.reshape(
                y_true,
                (-1, *grid_shape, 1, 5 + class_num)) # N*S*S*1*5+C
            y_pred = tf.reshape(
                y_pred,
                (-1, *grid_shape, bbox_num, 5 + class_num)) # N*S*S*B*5+C

            has_obj_mask = y_true[..., 4] # N*S*S*1

            pi_true = tf.argmax(y_true[..., -class_num:], # N*S*S*1*C
                                axis=-1) # N*S*S*1
            pi_pred = tf.argmax(y_pred[..., -class_num:], # N*S*S*B*C
                                axis=-1) # N*S*S*B
            
            equal_mask = tf.cast(pi_true == pi_pred,
                                dtype=y_true.dtype) # N*S*S*B
            equal_mask = equal_mask*has_obj_mask # N*S*S*B

            num_p = tf.reduce_sum(has_obj_mask)*bbox_num

            return tf.reduce_sum(equal_mask)/(num_p + epsilon)
        return class_acc


        
    '''
    Model Create
    '''
    def create_model(self,
                     anchors=[[0.89663461, 0.78365384],
                              [0.37500000, 0.47596153],
                              [0.27884615, 0.21634615],
                              [0.14182692, 0.28605769],
                              [0.14903846, 0.10817307],
                              [0.07211538, 0.14663461],
                              [0.07932692, 0.05528846],
                              [0.03846153, 0.07211538],
                              [0.02403846, 0.03125000]],
                     backbone="full_darknet",
                     pretrained_weights=None,
                     pretrained_darknet=None):
        
        if isinstance(pretrained_darknet, str):
            pre_body_weights = pretrained_darknet
            pretrained_darknet = None
        else:
            pre_body_weights = None
        
        model_body = self.yolo_neck(self.input_shape,
            pretrained_weights=pre_body_weights)

        if pretrained_darknet is not None:
            model_body.set_weights(pretrained_darknet.get_weights())
        self.model = self.yolo_head(model_body,
                               self.class_num,
                               anchors)
         
        if pretrained_weights is not None:
            self.model.load_weights(pretrained_weights)
        self.anchors = anchors
        self.grid_shape = self.model.output[0].shape[1:3]
        self.fpn_layers = len(self.model.output)
        self.abox_num = len(self.anchors)//self.fpn_layers

    '''
    Loss Create
    '''
    def loss(self,
             binary_weight=1,
             loss_weight=[1, 1, 5, 1],
             ignore_thresh=0.6):

        if (not isinstance(binary_weight, Iterable)
            or len(binary_weight) != self.fpn_layers):
            binary_weight = [binary_weight]*self.fpn_layers
        
        if isinstance(loss_weight, dict):
            loss_weight_list = []
            loss_weight_list.append(loss_weight["xy"])
            loss_weight_list.append(loss_weight["wh"])
            loss_weight_list.append(loss_weight["conf"])
            loss_weight_list.append(loss_weight["prob"])
            loss_weight = loss_weight_list
        
        loss_list = []
        for fpn_id in range(self.fpn_layers):
            grid_amp = 2**(fpn_id)
            grid_shape = (self.grid_shape[0]*grid_amp,
                          self.grid_shape[1]*grid_amp)
            anchors_id = self.abox_num*fpn_id
            loss_list.append(self.wrap_yolo_loss(
                grid_shape=grid_shape,
                bbox_num=self.abox_num, 
                class_num=self.class_num,
                anchors=self.anchors[
                    anchors_id:anchors_id + self.abox_num],
                binary_weight=binary_weight[fpn_id],
                loss_weight=loss_weight,
                ignore_thresh=ignore_thresh))
        return loss_list
    
    '''
    Metrics Create
    '''
    def metrics(self, type="obj_acc"):
        
        metrics_list = [[] for _ in range(self.fpn_layers)]
        for fpn_id in range(self.fpn_layers):
            grid_amp = 2**(fpn_id)
            grid_shape = (self.grid_shape[0]*grid_amp,
                            self.grid_shape[1]*grid_amp)
            
            if "obj" in type:
                metrics_list[fpn_id].append(
                    self.wrap_obj_acc(
                        grid_shape,
                        self.abox_num, 
                        self.class_num))
            if "iou" in type:
                metrics_list[fpn_id].append(
                    self.wrap_mean_iou(
                        grid_shape,
                        self.abox_num, 
                        self.class_num))
            if "class" in type:
                metrics_list[fpn_id].append(
                    self.wrap_class_acc(
                        grid_shape,
                        self.abox_num, 
                        self.class_num))
        
        return metrics_list


############## Yolov4 #################
############# Build Model Yolov4 #############
def Model_Yolov4(train_img, train_label, val_img, val_label, class_namesm, anchors):
    optimizer = Adam(learning_rate=1e-4)
    callback = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True) 

    yolo = Yolov4(class_names=class_names)
    yolo.create_model(anchors=anchors)

    binary_weight_list = []

    for i in range(len(train_label)):
        binary_weight_list.append(
            get_class_weight(
            train_label[i][..., 4:5],
            method='binary'
            )
        )

    binary_weight_list = [0.1]*3


    ignore_thresh = 0.7
    use_focal_loss = True

    loss_weight = {
        "xy":1,
        "wh":1,
        "conf":5,
        "prob":1
        }

    loss_fn = yolo.loss(
        binary_weight_list,
        loss_weight=loss_weight,
        ignore_thresh=ignore_thresh
        )
    metrics = yolo.metrics("obj+iou+class")

    yolo.model.compile(
        optimizer = optimizer,
        #optimizer=SGD(learning_rate=1e-10, momentum=0.9, decay=5e-4),
        loss = loss_fn,
        metrics = metrics
        )
    
    train_history = yolo.model.fit(train_img,
                                    train_label,
                                    epochs = n_epoch,
                                    batch_size=10,
                                    verbose=1,
                                    validation_data=(val_img, val_label),
                                    callbacks=[callback]
                                    )
    return yolo
############# Build Model Yolov4 #############

def cal_iou_v2(xywh_true, xywh_pred):
    """Calculate IOU of two tensors.

    Args:
        xywh_true: A tensor or array-like of shape (..., 4).
            (x, y) should be normalized by image size.
        xywh_pred: A tensor or array-like of shape (..., 4).
    Returns:
        An iou_scores array.
    """
    xy_true = xywh_true[..., 0:2] # N*1*1*1*(S*S)*2
    wh_true = xywh_true[..., 2:4]

    xy_pred = xywh_pred[..., 0:2] # N*S*S*B*1*2
    wh_pred = xywh_pred[..., 2:4]
    
    half_xy_true = wh_true / 2.
    mins_true    = xy_true - half_xy_true
    maxes_true   = xy_true + half_xy_true

    half_xy_pred = wh_pred / 2.
    mins_pred    = xy_pred - half_xy_pred
    maxes_pred   = xy_pred + half_xy_pred       
    
    intersect_mins  = np.maximum(mins_pred,  mins_true)
    intersect_maxes = np.minimum(maxes_pred, maxes_true)
    intersect_wh    = np.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]
    
    true_areas = wh_true[..., 0] * wh_true[..., 1]
    pred_areas = wh_pred[..., 0] * wh_pred[..., 1]

    union_areas = pred_areas + true_areas - intersect_areas
    iou_scores  = intersect_areas/(union_areas + epsilon)
    
    return iou_scores

def CreateAnchors(img, label, cluster, type = "1_dimension"):
    if type == '1_dimension':
        all_boxes = label[label[..., 4] == 1][..., 2:4]
        anchors = kmeans(
            all_boxes,
            n_cluster=int(cluster),
            dist_func=iou_dist,
            stop_dist=0.00001)

        anchors = np.sort(anchors, axis=0)[::-1]
    else:
        all_boxes = train_label[2][train_label[2][..., 4] == 1][..., 2:4]
        anchors = kmeans(
            all_boxes,
            n_cluster=int(cluster),
            dist_func=iou_dist,
            stop_dist=0.000001)

        anchors = np.sort(anchors, axis=0)[::-1]

    return anchors, all_boxes

def cal_iou_nms(xywh_true, xywh_pred):
    """Calculate IOU of two tensors.

    Args:
        xywh_true: A tensor or array-like of shape (..., 4).
            (x, y) should be normalized by image size.
        xywh_pred: A tensor or array-like of shape (..., 4).
    Returns:
        An iou_scores array.
    """
    xy_true = xywh_true[..., 0:2] # N*1*1*1*(S*S)*2
    wh_true = xywh_true[..., 2:4]

    xy_pred = xywh_pred[..., 0:2] # N*S*S*B*1*2
    wh_pred = xywh_pred[..., 2:4]
    
    half_xy_true = wh_true / 2.
    mins_true    = xy_true - half_xy_true
    maxes_true   = xy_true + half_xy_true

    half_xy_pred = wh_pred / 2.
    mins_pred    = xy_pred - half_xy_pred
    maxes_pred   = xy_pred + half_xy_pred       
    
    intersect_mins  = np.maximum(mins_pred,  mins_true)
    intersect_maxes = np.minimum(maxes_pred, maxes_true)
    intersect_wh    = np.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]
    
    true_areas = wh_true[..., 0] * wh_true[..., 1]
    pred_areas = wh_pred[..., 0] * wh_pred[..., 1]

    union_areas = pred_areas + true_areas - intersect_areas
    iou_scores  = intersect_areas/(union_areas + epsilon)
    
    return iou_scores

def soft_nms(xywhcp, class_num=1,
        nms_threshold=0.5, conf_threshold=0.5, sigma=0.5, version = 2):
    """Soft Non-Maximum Suppression.

    Args:
        xywhcp: output from `decode()`.
        class_num:  An integer,
            number of classes.
        nms_threshold: A float, default is 0.5.
        conf_threshold: A float,
            threshold for quantizing output.
        sigma: A float,
            sigma for Soft NMS.

    Returns:
        xywhcp through nms.
    """
    argmax_prob = xywhcp[..., 5].astype("int")

    xywhcp_new = []
    for i_class in range(class_num):
        xywhcp_class = xywhcp[argmax_prob==i_class]
        xywhc_class = xywhcp_class[..., :5]
        prob_class = xywhcp_class[..., 6]

        xywhc_axis0 = np.reshape(
            xywhc_class, (-1, 1, 5))
        xywhc_axis1 = np.reshape(
            xywhc_class, (1, -1, 5))

        iou_scores = cal_iou_nms(xywhc_axis0, xywhc_axis1)
        # conf = xywhc_class[..., 4]*prob_class
        conf = xywhc_class[..., 4]
        sort_index = np.argsort(conf)[::-1]

        white_list = []
        delete_list = []
        for conf_index in sort_index:
            white_list.append(conf_index)
            iou_score = iou_scores[conf_index]
            overlap_indexes = np.where(iou_score >= nms_threshold)[0]

            for overlap_index in overlap_indexes:
                if overlap_index not in white_list:
                    conf_decay = np.exp(-1*(iou_score[overlap_index]**2)/sigma)
                    conf[overlap_index] *= conf_decay
                    if conf[overlap_index] < conf_threshold:
                        delete_list.append(overlap_index)
        xywhcp_class = np.delete(xywhcp_class, delete_list, axis=0)
        xywhcp_new.append(xywhcp_class)
    if version == 3:
        xywhcp_new = sorted(xywhcp_new[0], reverse=True, key=lambda x:x[4])
    if version == 4:
        xywhcp_new = sorted(xywhcp_new[0], reverse=True, key=lambda x:x[3])
    xywhcp = np.vstack(xywhcp_new)
    return xywhcp


def nms(xywhcp, class_num=1, nms_threshold=0.5):
    """Non-Maximum Suppression.

    Args:
        xywhcp: output from `decode()`.
        class_num:  An integer,
            number of classes.
        nms_threshold: A float, default is 0.5.

    Returns:
        xywhcp through nms.
    """
    argmax_prob = xywhcp[..., 5].astype("int")

    xywhcp_new = []
    for i_class in range(class_num):
        xywhcp_class = xywhcp[argmax_prob==i_class]
        xywhc_class = xywhcp_class[..., :5]
        prob_class = xywhcp_class[..., 6]

        xywhc_axis0 = np.reshape(
            xywhc_class, (-1, 1, 5))
        xywhc_axis1 = np.reshape(
            xywhc_class, (1, -1, 5))

        iou_scores = cal_iou_v2(xywhc_axis0, xywhc_axis1)
        conf = xywhc_class[..., 4]*prob_class
        sort_index = np.argsort(conf)[::-1]

        white_list = []
        delete_list = []
        for conf_index in sort_index:
            white_list.append(conf_index)
            if conf_index not in delete_list:
                iou_score = iou_scores[conf_index]
                overlap_indexes = np.where(iou_score >= nms_threshold)[0]

                for overlap_index in overlap_indexes:
                    if overlap_index not in white_list:
                        delete_list.append(overlap_index)
        xywhcp_class = np.delete(xywhcp_class, delete_list, axis=0)
        xywhcp_new.append(xywhcp_class)
    
    xywhcp_new = sorted(xywhcp_new, reverse=True, key=lambda x:x[0][5])
    st.write(xywhcp_new)
    # xywhcp_new_list = xywhcp_new[0].tolist()

    # print(xywhcp_new[0])
    #### stage 2: loop over all boxes, remove boxes with high IOU
    xywhcp_final = []
    # while(len(xywhcp_new) > 0):

    ### Này code chữa cháy thôi á chừng nào nhiều hơn 1 đối tượng thì phải code khác
    while(len(xywhcp_new) > 0):
        # print("vô đây ra mà")
        current_box = xywhcp_new.pop(0)
        index = np.argmax(current_box[:, 4])  # Cột thứ 5 (index 4) chứa giá trị cần tìm

        # Lấy mảng có giá trị cao nhất
        max_array = current_box[index]
        xywhcp_final.append(max_array)
        # print(current_box)
        # for box in xywhcp_new:
        #     if( current_box[5] == box[5]):
        #         print(current_box[..., 0:2])
                # print(box[..., 3:4])
                # iou = cal_iou_v2(current_box[:4], box[:4])
                # st.write(iou)
                # if(iou > 0.4):
                #    xywhcp_new_list[0].remove(box)

    xywhcp = np.vstack(xywhcp_final)
    return xywhcp



def decode(*label_datas,
           class_num=1,
           threshold=0.5,
           version=1):
    """Decode the prediction from yolo model.

    Args:
        *label_datas: Ndarrays,
            shape: (grid_heights, grid_widths, info).
            Multiple label data can be given at once.
        class_num:  An integer,
            number of classes.
        threshold: A float,
            threshold for quantizing output.
        version: An integer,
            specifying the decode method, yolov1、v2 or v3.

    Return:
        Numpy.ndarray with shape: (N, 7).
            7 values represent:
            x, y, w, h, c, class index, class probability.
    """
    output = []
    for label_data in label_datas:
        grid_shape = label_data.shape[:2]
        if version == 1:
            bbox_num = (label_data.shape[-1] - class_num)//5
            xywhc = np.reshape(label_data[..., :-class_num],
                               (*grid_shape, bbox_num, 5))
            prob = np.expand_dims(
                label_data[..., -class_num:], axis=-2)
        elif version == 2 or version == 3:
            bbox_num = label_data.shape[-1]//(5 + class_num)
            label_data = np.reshape(label_data,
                                    (*grid_shape,
                                     bbox_num, 5 + class_num))
            xywhc = label_data[..., :5]
            prob = label_data[..., -class_num:]
        else:
            raise ValueError("Invalid version: %s" % version)   

        # joint_conf = xywhc[..., 4:5]*prob
        joint_conf = xywhc[..., 4:5]
        where = np.where(joint_conf >= threshold)

        for i in range(len(where[0])):
            x_i = where[1][i]
            y_i = where[0][i]
            box_i = where[2][i]
            class_i = where[3][i]

            x_reg = xywhc[y_i, x_i, box_i, 0]
            y_reg = xywhc[y_i, x_i, box_i, 1]
            w_reg = xywhc[y_i, x_i, box_i, 2]
            h_reg = xywhc[y_i, x_i, box_i, 3]
            conf = xywhc[y_i, x_i, box_i, 4]

            x = (x_i + x_reg)/grid_shape[1]
            y = (y_i + y_reg)/grid_shape[0]
            
            w = w_reg
            h = h_reg
            
            if version == 1:
                p = prob[y_i, x_i, 0, class_i]
            else:
                p = prob[y_i, x_i, box_i, class_i]
            output.append([x, y, w, h, conf, class_i, p])
    output = np.array(output, dtype="float")
    return output



###--------------------------------------- Model CNN -----------------------------------------------###
def CNN_Model():
    num_classes = 36
    
    # Sử dụng ResNet50 thay thế cho VGG16
    # base_model = ResNetRS420(weights='imagenet', include_top=False, input_shape=(100, 75, 3))
    base_model = ResNetRS420(include_top=False, input_shape=(100, 75, 3))

    for layer in base_model.layers:
        layer.trainable = False

    x = Flatten()(base_model.output)
    x = Dense(256, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

###--------------------------------------- Model CNN -----------------------------------------------###

###--------------------------------------- GFPGAN----------------------------------------------------###

# from GFPGAN.gfpgan import GFPGANer
# if "imported_GFPGANer" not in st.session_state:
#     print("aloooooooooooo")
#     from GFPGAN.gfpgan import GFPGANer
#     st.session_state.imported_GFPGANer = True
from GFPGAN.gfpgan.utils import GFPGANer

def func_GFPGAN(input_img, bg_upsampler = 'realesrgan', bg_tile = 400, version = '1.3', upscale = 2, weight = 0.5):
    if bg_upsampler == 'realesrgan':
        if not torch.cuda.is_available():  # CPU
            # import warnings
            # warnings.warn('The unoptimized RealESRGAN is slow on CPU. We do not use it. '
            #               'If you really want to use it, please modify the corresponding codes.')
            # bg_upsampler = None
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
            bg_upsampler = RealESRGANer(
                scale=2,
                model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth',
                model=model,
                tile=bg_tile,
                tile_pad=10,
                pre_pad=0,
                half=False)  # need to set False in CPU mode
        else:
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
            bg_upsampler = RealESRGANer(
                scale=2,
                model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth',
                model=model,
                tile=bg_tile,
                tile_pad=10,
                pre_pad=0,
                half=True)  # need to set False in CPU mode
    else:
        bg_upsampler = None


    # ------------------------ set up GFPGAN restorer ------------------------
    if version == '1':
        arch = 'original'
        channel_multiplier = 1
        model_name = 'GFPGANv1'
        url = 'https://github.com/TencentARC/GFPGAN/releases/download/v0.1.0/GFPGANv1.pth'
    elif version == '1.2':
        arch = 'clean'
        channel_multiplier = 2
        model_name = 'GFPGANCleanv1-NoCE-C2'
        url = 'https://github.com/TencentARC/GFPGAN/releases/download/v0.2.0/GFPGANCleanv1-NoCE-C2.pth'
    elif version == '1.3':
        arch = 'clean'
        channel_multiplier = 2
        model_name = 'GFPGANv1.3'
        url = 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth'
    elif version == '1.4':
        arch = 'clean'
        channel_multiplier = 2
        model_name = 'GFPGANv1.4'
        url = 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth'
    elif version == 'RestoreFormer':
        arch = 'RestoreFormer'
        channel_multiplier = 2
        model_name = 'RestoreFormer'
        url = 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/RestoreFormer.pth'
    else:
        raise ValueError(f'Wrong model version {version}.')
    

    # determine model paths
    model_path = os.path.join('GFPGAN/experiments/pretrained_models', model_name + '.pth')
    if not os.path.isfile(model_path):
        model_path = os.path.join('gfpgan/weights', model_name + '.pth')
    if not os.path.isfile(model_path):
        # download pre-trained models from url
        model_path = url


    restorer = GFPGANer(
        model_path=model_path,
        upscale=upscale,
        arch=arch,
        channel_multiplier=channel_multiplier,
        bg_upsampler=bg_upsampler)
    

    # restore faces and background if necessary
    cropped_faces, restored_faces, restored_img = restorer.enhance(
        input_img,
        # has_aligned=args.aligned,
        # only_center_face=args.only_center_face,
        paste_back=True,
        weight=weight)
    
    restorer = None
    return restored_img

###--------------------------------------- GFPGAN----------------------------------------------------###




async def load_Yolov2():
    yolo = Yolov2(class_names=class_names)
    # anchors=[[0.18855932, 0.26236546],
    #         [0.17818911, 0.24526663],
    #         [0.17066398, 0.23265725],
    #         [0.16398123, 0.22219862],
    #         [0.15771288, 0.21271414],
    #         [0.15161675, 0.2030504 ],
    #         [0.14446504, 0.1919692 ],
    #         [0.13439809, 0.17940256],
    #         [0.1279203 , 0.16920881],
    #         [0.1223517 , 0.15594059]]
    anchors = [[0.17906713, 0.24645409],
            [0.16138585, 0.21843861],
            [0.13676852, 0.18154225]]
    yolo.create_model(anchors=anchors)
    yolo.model.load_weights('./Weights/yolov2_3_anchor.h5')
    return yolo

async def load_Yolov3():
    yolo = Yolov3(class_names=class_names)
    # anchors=[[0.18779223, 0.26137096],
    #         [0.17727496, 0.24374145],
    #         [0.16971926, 0.2308511 ],
    #         [0.16248433, 0.21979941],
    #         [0.1547918 , 0.20906165],
    #         [0.14827792, 0.19636963],
    #         [0.13722295, 0.18287213],
    #         [0.12978053, 0.17237878],
    #         [0.12378935, 0.16336633]]
    anchors=[[0.17974663, 0.24750794],
            [0.16224557, 0.21994163],
            [0.13819972, 0.18308146]]
    yolo.create_model(anchors=anchors)
    yolo.model.load_weights('./Weights/yolov3_3_anchor.h5')
    return yolo


async def load_Yolov4():
    yolo = Yolov4(class_names=class_names)
    # anchors=[[0.26923078, 0.26923078],
    #         [0.20192307, 0.20192307],
    #         [0.19016773, 0.13461539],
    #         [0.13461539, 0.10096154],
    #         [0.10120743, 0.0673077 ],
    #         [0.10096154, 0.06188303],
    #         [0.0673077 , 0.05288462],
    #         [0.05288462, 0.05113026],
    #         [0.03365385, 0.03365385]]

    anchors=[[0.17842099, 0.24545719],
            [0.16058964, 0.21712816],
            [0.13592426, 0.18043523]]

    # anchors=[[0.1688862 , 0.22677982],
    #         [0.14512712, 0.18811882],
    #         [0.1292373 , 0.16831683]]
    yolo.create_model(anchors=anchors)
    yolo.model.load_weights('./Weights/yolov4_mish_3_anchor.h5')
    return yolo

async def load_CNN():
    model_cnn=CNN_Model()
    model_cnn.load_weights("./Weights/cnn_resnetrs420_epochs_7.h5")
    return model_cnn


async def run_Yolov2andCNN():
    yolo, cnn = await asyncio.gather(
        load_Yolov2(), 
        load_CNN()
    )
    return yolo, cnn
async def run_Yolov3andCNN():
    yolo, cnn = await asyncio.gather(
        load_Yolov3(), 
        load_CNN()
    )
    return yolo, cnn
async def run_Yolov4andCNN():
    yolo, cnn = await asyncio.gather(
        load_Yolov4(), 
        load_CNN()
    )
    return yolo, cnn

async def run_Yolov2():
    yolo = await asyncio.gather(
        load_Yolov2()
    )
    return yolo
async def run_Yolov3():
    yolo = await asyncio.gather(
        load_Yolov3()
    )
    return yolo
async def run_Yolov4():
    yolo = await asyncio.gather(
        load_Yolov4()
    )
    return yolo

##################################   Yolov8     ###########################
from dataset.build import build_transform

from utils.misc import load_weight
from utils.box_ops import rescale_bboxes

from models.detectors import build_model
from config import build_dataset_config, build_trans_config, build_model_config

class Args():
    def __init__(self):
        self.img_size = 640
        self.mosaic = None
        self.mixup = None
        self.mode = 'image'
        self.cuda = False
        self.show = False
        self.gif = False
        # Model setting
        self.model = 'yolov8_n'
        self.num_classes = 1
        self.weight = './Weights/yolov8_n_last_mosaic_epoch.pth'
        self.conf_thresh = 0.35
        self.nms_thresh = 0.5
        self.topk = 100
        self.deploy = False
        self.fuse_conv_bn = False
        self.no_multi_labels = False
        self.nms_class_agnostic = False
        # Data Setting
        self.dataset = 'plate_number'
async def load_Yolov8():
    args = Args()
    if args.cuda:
        print('use cuda')
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # config
    model_cfg = build_model_config(args)
    trans_cfg = build_trans_config(model_cfg['trans_type'])
    data_cfg  = build_dataset_config(args)

    ## Data info
    num_classes = data_cfg['num_classes']
    class_names = data_cfg['class_names']
    class_indexs = data_cfg['class_indexs']

    # build model
    model = build_model(args, model_cfg, device, num_classes, False)

    model = load_weight(model, args.weight, args.fuse_conv_bn)
    model.to(device).eval() 
    
    return model

##################################   Yolov8     ###########################

async def load_all_model():
    yolov2, yolov3, yolov4, yolov8, cnn = await asyncio.gather(
        load_Yolov2(),
        load_Yolov3(),
        load_Yolov4(),
        load_Yolov8(),
        load_CNN()
    )
    return yolov2, yolov3, yolov4, yolov8, cnn
async def load_custom_model():
    yolov3, yolov4, cnn = await asyncio.gather(
        load_Yolov3(),
        load_Yolov4(),
        load_CNN()
    )
    return yolov3, yolov4, cnn



if 'cnn' not in st.session_state:
    yolov2, yolov3, yolov4, yolov8, cnn = asyncio.run(load_all_model())
    st.session_state['yolov2'] = yolov2
    st.session_state['yolov3'] = yolov3
    st.session_state['yolov4'] = yolov4
    st.session_state['yolov8'] = yolov8
    st.session_state['cnn'] = cnn
# if 'cnn' not in st.session_state:
#     yolov3, yolov4, cnn = asyncio.run(load_custom_model())
#     st.session_state['yolov3'] = yolov3
#     st.session_state['yolov4'] = yolov4
#     st.session_state['cnn'] = cnn


from st_pages import Page, Section, add_page_title, show_pages

show_pages(
    [
        Page("home.py", "Trang Chủ", "🏠"),
        Page("train.py", "Train Model", "🧰"),
        Page("test.py", "Test Model", "🧰"),

        Section(name="Yolo V2", icon="👀"),
        # The pages appear in the order you pass them
        Page("V2/full_demo.py", "Full Demo With YoloV2", "📖"),
        Page("V2/detail_demo.py", "Detail Demo", "📖"),

        Section(name="Yolo V3", icon="👀"),
        # The pages appear in the order you pass them
        Page("V3/full_demo.py", "Full Demo With YoloV3", "📖"),
        Page("V3/detail_demo.py", "Detail Demo", "📖"),

        Section(name="Yolo V4", icon="👀"),
        # The pages appear in the order you pass them
        Page("V4/full_demo.py", "Full Demo With YoloV4", "📖"),
        Page("V4/detail_demo.py", "Detail Demo", "📖"),
        Section(name="Yolo V8", icon="👀"),
        # The pages appear in the order you pass them
        Page("V8/full_demo.py", "Full Demo With YoloV8", "📖"),
        # Page("V4/detail_demo.py", "Detail Demo", "📖"),
    ]
)
add_page_title()
s = f"<h1 style='text-align: center; color:#ED8C02; font-size:70px;'>Khóa Luận Tốt Nghiệp</h1>"
st.markdown(s, unsafe_allow_html=True) 
s = f"<h1 style='text-align: center;'>Xây Dựng Hệ Thống Nhận Diện Biển Số Xe Bằng Mô Hình Học Sâu</h1>"
st.markdown(s, unsafe_allow_html=True) 
s = f"<p style='text-align: right; font-size:30px;'>GVHD: TS Nguyễn Thành Sơn</p>"
st.markdown(s, unsafe_allow_html=True)  
s = f"<p style='text-align: right; font-size:30px;'>20133034 - Lê Minh Đăng</p>"
st.markdown(s, unsafe_allow_html=True) 
s = f"<p style='text-align: right; font-size:30px;'>20133075 - Võ Hoàng Nguyên</p>"
st.markdown(s, unsafe_allow_html=True) 