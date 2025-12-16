'''
Script for contrast synthesis using a CL-pretrained model
3D network changes
July 7: 3D network with downsampling along z to allow for partial decoder to work with 4x4x8 patches
July 27: Modifying scripts for T1-Gd synthesis from T1w and T2-FLAIR images.
'''

import sys
import os
import numpy as np
import pathlib
import logging
from keras.callbacks import ModelCheckpoint

 
# sys.path.append('/gpfs/scratch/pa2297/CCL-Synthetis/')
sys.path.append("/gpfs/scratch/sa9063/sakina-thesis/CCL-Synthetis/CCL-Synthetis/")

# import argparse
# parser = argparse.ArgumentParser()

# parser.add_argument('--is_baseline', type=int, default=0, choices=[0,1])
# parser.add_argument('--is_CL', type=int, default=0, choices=[0,1])
# parser.add_argument('--gpus_available', type=str, default='0')
# parser.add_argument('--no_of_tr_imgs', type=str, default='tr10')
# parser.add_argument('--data_aug', type=int, default=1, choices=[0,1])
# # parser.add_argument('--run_num', type=str, default='c3', choices=['c1','c2','c3','c4','c5','c6'])
# parser.add_argument('--lr_finetune', type=float, default=1e-3)
# parser.add_argument('--datatype', type=str, default='t1ce_flair')

# user_cfg = parser.parse_args()

 
from Synthesis import synth_config as cfg

# if not (user_cfg.is_baseline or user_cfg.is_CL):
#     exit('Please select one of the following: baseline or CL pretrained')
# else:
#     print('Starting training')
#     cfg.is_baseline = user_cfg.is_baseline
#     cfg.is_CL = user_cfg.is_CL
#     cfg.gpus_available = user_cfg.gpus_available
#     cfg.no_of_tr_imgs = user_cfg.no_of_tr_imgs
#     cfg.data_aug = user_cfg.data_aug
#     cfg.run_num = user_cfg.run_num
#     cfg.lr_finetune = user_cfg.lr_finetune

os.environ['CUDA_VISIBLE_DEVICES'] = cfg.gpus_available
# Suppressing TF message printouts
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)
import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

from utils.utils import *


opShape = (cfg.img_size_x,cfg.img_size_y) 
cfg.num_channels = len(cfg.contrast_idx)

 
''' Load model architecture '''
from utils.model_utils import modelObj
model = modelObj(cfg)

from Synthesis.synthesis_losses import lossObj
loss = lossObj(cfg)
 


# Load pretrained wts

if cfg.is_baseline:
    print('Baseline model')
    pretrained_model_weights = None
elif cfg.is_CL:
    print('Contrastive loss with CL pretrained model')
    pretrained_model_weights = cfg.CL_param_wts 
else:
    print('Not supported')

# Training the model
print('Training RegUNET')
ae_reg = model.synth_unet()               # act_name = 'tanh' default is 'ReLU'
if pretrained_model_weights is not None: 
    print('Loading weights from: ', pretrained_model_weights)
    ae_reg.load_weights(pretrained_model_weights, by_name=True)

# Choose an appropriate loss function
print('Creating custom loss function for', cfg.loss_function)
if cfg.loss_function == 'L1':
    customLoss = 'mae'
elif cfg.loss_function == 'L2':
    customLoss = 'mse'
elif cfg.loss_function == 'ssim':
    customLoss = loss.SSIMLoss()
else:
    print('Instantiating Vgg16 feature extractor')
    vgg_model = model.vgg16_feature_extractor()       
    customLoss = loss.custom_perceptualLoss(vgg_model)

if cfg.optimizer == 'SGD':
    Adamopt = tf.keras.optimizers.SGD(learning_rate=cfg.lr_finetune, momentum=.9) 
else:           
    Adamopt = tf.keras.optimizers.Adam(learning_rate=cfg.lr_finetune) 


ae_reg.compile(loss=customLoss, optimizer=Adamopt)


# Create save directory
save_dir = cfg.save_dir + cfg.test_dir

if cfg.is_baseline:
    save_dir = save_dir + '_baseline'
else:
    save_dir = save_dir + '_CL_' + cfg.start_str  

save_dir = save_dir + '/' + cfg.datatype + '_' + cfg.no_of_tr_imgs + '_' + cfg.loss_function
if cfg.data_aug:
    save_dir = save_dir + '_data_aug'
save_dir = save_dir + '/tr_comb_' + str(cfg.run_num) + '_LR_ft_'
save_dir = save_dir + str(cfg.lr_finetune) + '/'

print('Creating', save_dir)

pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)
modelSavePath = save_dir + 'weights_{epoch:02d}.hdf5'
csvPath = save_dir + 'training.log'
callbacks = get_callbacks(csvPath)
callbacks.append(ModelCheckpoint(
    filepath=modelSavePath,
    save_weights_only=True,
    save_best_only=False,
    verbose=1
))

# Load the data generator
from Datagen.h5_pretrain_Synth_Data_Generator import DataLoaderObj

train_gen = DataLoaderObj(cfg, 
                        train_flag=True) 

val_gen = DataLoaderObj(cfg, 
                        train_flag=False) 

# Train the model
initial_epoch = 0
num_epochs = cfg.num_epochs 
history = ae_reg.fit(train_gen,
                        epochs=num_epochs,
                        verbose=1,
                        callbacks=callbacks,
                        validation_data=val_gen,
                        initial_epoch=initial_epoch,
                        use_multiprocessing=False)


# Save final weights
ae_reg.save_weights(
    save_dir + 'weights_' + str
    (num_epochs) + 'Final' +'.hdf5',
    overwrite=True,
    save_format='h5',
    options=None)

  
# Save the config file with run parameters
cfg_txt_name = save_dir + 'config_params.txt'
with open(cfg_txt_name, 'w') as f:
    for name, value in cfg.__dict__.items():
        f.write('{} = {!r}\n'.format(name, value)) 
        
