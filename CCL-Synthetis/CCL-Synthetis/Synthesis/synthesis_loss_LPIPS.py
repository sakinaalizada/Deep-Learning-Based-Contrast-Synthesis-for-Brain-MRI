#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2D loss for contrast synthesis
Created on Mon Apr  4 12:20:34 2022

@author: lavanya
"""
 
import tensorflow as tf
import numpy as np

class lossObj:
    def __init__(self, cfg):
        print('Loss init')
        self.cfg = cfg
        self.op_channels = len(cfg.target_contrast_idx)
        self.slc_select = 4
        self.batch_size = cfg.batch_size
        self.perceptualLoss_weight = cfg.perceptualLoss_weight # scalar factor for importance of mae vs perceptual loss
        # self.mae_weight = cfg.mae_weight 
    
    '''Edited by Parisima'''
    # SSIM as Loss function
    def SSIMLoss(self):
        def recon_SSIMLoss(y_true, y_pred):
            temp_y_true = y_true[..., 0]
            temp_y_pred = y_pred[..., 0]
            # Add a channel dimension
            temp_y_true = tf.expand_dims(temp_y_true, axis=-1)
            temp_y_pred = tf.expand_dims(temp_y_pred, axis=-1)

            # Scale images to [0, 1]
            y_true_rescaled = (temp_y_true - tf.reduce_min(temp_y_true)) / (tf.reduce_max(temp_y_true) - tf.reduce_min(temp_y_true))
            y_pred_rescaled = (temp_y_pred - tf.reduce_min(temp_y_pred)) / (tf.reduce_max(temp_y_pred) - tf.reduce_min(temp_y_pred))

            return 1 - tf.reduce_mean(tf.image.ssim(y_true_rescaled, y_pred_rescaled, 1.0))
        return recon_SSIMLoss


    # Custom reconstruction loss using a feature extractor style model
    def custom_perceptualLoss(self, styleModel):
        def Recon_perceptualLoss(y_true, y_pred):   
            # Calculate loss per output channel for VGG compatibility
            batch_loss = 0.0
            mse = tf.keras.losses.MeanSquaredError()
            '''Edited by Parisima'''
            # mae = tf.keras.losses.MeanAbsoluteError()

            # output is batch x height x width x channels
            for channel_idx in range(self.op_channels):
                temp_y_true = y_true[...,channel_idx]
                temp_y_pred = y_pred[...,channel_idx]
                temp_y_true = tf.expand_dims(temp_y_true, axis=-1)
                temp_y_pred = tf.expand_dims(temp_y_pred, axis=-1)

                '''Edited by Parisima'''
                # Calculate MAE
                # mae_loss = mae(temp_y_true, temp_y_pred)

                # RGB channels for vgg compatibility
                y_true_vgg = tf.concat([temp_y_true,temp_y_true,temp_y_true],axis=-1)      
                y_pred_vgg = tf.concat([temp_y_pred,temp_y_pred,temp_y_pred],axis=-1)        
                # Get the feature maps from VGG16
                style_fts_true = styleModel(y_true_vgg)     
                style_fts_pred = styleModel(y_pred_vgg)
                _,H,W,CH = style_fts_true.get_shape().as_list()
                if H is None or W is None or CH is None:
                    scalar_mult = 1.0
                else:
                    scalar_mult = 1/ ( H * W * CH)
                style_loss = scalar_mult * tf.reduce_mean(mse(style_fts_true, style_fts_pred))
                '''Edited by Parisima'''
                # combined_loss = (self.mae_weight * mae_loss) + (self.perceptualLoss_weight * style_loss)
                # batch_loss += combined_loss
                batch_loss = batch_loss +  (self.perceptualLoss_weight * style_loss) 
            total_loss = batch_loss 
            # total_loss = batch_loss / self.op_channels
            return tf.reduce_mean(total_loss)
        return Recon_perceptualLoss   
    
 