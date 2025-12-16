#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 18:37:59 2022

@author: lavanya
For T2, T1 synthesis using T1ce + T2-FLAIR

"""
gpus_available = '0'

import os, logging
os.environ['CUDA_VISIBLE_DEVICES'] = gpus_available
# Suppressing TF message printouts
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)

''' Finetuning params'''

is_baseline = 0
is_CL = 1

''' CL params '''
full_decoder = 1
partial_decoder = 0
freeze_pretrained = 0

''' 
Select MP-MR contrasts for processing
0: T1-Gd
1: T2-weighted
2: T1-weighted
3: T2-FLAIR
'''

contrast_idx = [1,2,3]
target_contrast_idx = [0]

datatype = 't1-t2-flair-50loss1'
run_num = 'Exp1'
no_of_tr_imgs = 'tr1'

batch_size = 8
lr_finetune = 1e-4

''' Data params'''
dataset = 'nyu_mets'
img_size_x = 160                      #**************************
img_size_y = 160                      #**************************
num_channels = len(contrast_idx) # input channel
num_output_channel = len(target_contrast_idx) # output channel
num_epochs = 100 # it was 200 before
control_size_of_dataset = 24                      #**************************

'''Model params'''
conv_kernel = (3,3,3)
no_filters = [1, 16, 32, 64, 128, 128]
latent_dim = 16
optimizer = 'Adam'
num_ft_channels = latent_dim  
loss_function = 'Hybride'  # options are L1, L2, L1+perceptual hybride / Hybride / ssim    #**************************

''' Augmentation params'''
data_aug = 0
augmentation_ratio = 0.5
rand_deform = 1
sigma = 4
mu = 0
rand_brit_cont = 0
# perceptualLoss_weight = 1
perceptualLoss_weight = 0.05
# mae_weight = 0.0005
mae_weight = 0.01
# mse_weight = 0.0005
mse_weight = 1
base_dir = '/gpfs/scratch/sa9063/sakina-thesis/outputs/'



save_str = dataset + '_' + datatype
save_dir = os.path.join(base_dir, save_str)

test_dir = '/finetune_nyumets_on_brats_finetuned_relu_hybride'       #**************************

''' Relevant directories '''
if rand_deform:
    save_dir = os.path.join(save_dir, 'rand_deform') #zmean_rand_deform
else:
    save_dir =  os.path.join(save_dir, '')


hdf5_train_img_filename = '/gpfs/scratch/sa9063/sakina-thesis/Results/train_t1-t2-flair_pretrain_img_zmean.hdf5'
#**************************
# hdf5_train_img_filename = '/gpfs/data/fenglab/ParisimaAbdali/pa2297/Training/t1-t2-flair/Datah5/Zmean350-synth/train_t1-t2-flair_pretrain_img_zmean.hdf5'
#**************************
hdf5_val_img_filename   = '/gpfs/scratch/sa9063/sakina-thesis/Results/val_t1-t2-flair_pretrain_img_zmean.hdf5'
#**************************
# hdf5_val_img_filename = '/gpfs/data/fenglab/ParisimaAbdali/pa2297/Training/t1-t2-flair/Datah5/Zmean350-synth/val_t1-t2-flair_pretrain_img_zmean.hdf5'


''' Pretrained_weights'''
if full_decoder:
    CL_param_wts = "/gpfs/data/fenglab/ParisimaAbdali/pa2297/pa2297/CCL-Thesis/RelU/Loss/brats_t1_t2_flair_loss1/rand_deform/finetune_Hybride _CL_partial_dec_warm/t1_t2_flair_loss1_tr1_Hybride/tr_comb_Exp1_LR_ft_0.01/weights_20Final.hdf5"
    # "/gpfs/data/fenglab/ParisimaAbdali/pa2297/Training/t1-t2-flair/RelU/pretrain_full_dec_zmean/t1-t2-flair/patchsize_4_LR0.001_tau0.1_top100/weights_150.hdf5"
    start_str = 'full_dec_warm'
elif partial_decoder:
    CL_param_wts = '/gpfs/scratch/pa2297/Training/t1-t2-flair/RelU/pretrain_partial_dec/t1-t2-flair/patchsize_4_LR0.001_tau0.1_top100/weights_150.hdf5'
    start_str = 'partial_dec_warm'
 
