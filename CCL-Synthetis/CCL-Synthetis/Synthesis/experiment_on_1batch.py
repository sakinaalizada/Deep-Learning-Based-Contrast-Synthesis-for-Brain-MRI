import sys
import os
import pathlib
import logging

sys.path.append("/gpfs/scratch/sa9063/sakina-thesis/CCL-Synthetis/CCL-Synthetis")

from Synthesis import synth_config as cfg
from Datagen.h5_pretrain_Synth_Data_Generator import DataLoaderObj
from utils.model_utils import modelObj
from Synthesis.synthesis_losses import lossObj
from utils.utils import get_callbacks

import tensorflow as tf


train_gen = DataLoaderObj(cfg, train_flag=True)
val_gen   = DataLoaderObj(cfg, train_flag=False)

x_tr, y_tr = train_gen[0]
x_val, y_val = val_gen[0]
print("Train batch:", x_tr.shape, y_tr.shape)
print("Val batch:",   x_val.shape, y_val.shape)

model_builder = modelObj(cfg)
ae_reg = model_builder.synth_unet()

loss_builder = lossObj(cfg)

print('Creating custom loss function for', cfg.loss_function)
if cfg.loss_function == 'L1':
    customLoss = 'mae'
elif cfg.loss_function == 'L2':
    customLoss = 'mse'
elif cfg.loss_function == 'ssim':
    customLoss = loss_builder.SSIMLoss()
else:
    print('Instantiating Vgg16 feature extractor')
    vgg_model = model_builder.vgg16_feature_extractor()
    customLoss = loss_builder.custom_perceptualLoss(vgg_model)

if getattr(cfg, "optimizer", "Adam") == 'SGD':
    Adamopt = tf.keras.optimizers.SGD(
        learning_rate=cfg.lr_finetune, momentum=.9
    )
else:
    Adamopt = tf.keras.optimizers.Adam(learning_rate=cfg.lr_finetune)

ae_reg.compile(loss=customLoss, optimizer=Adamopt)

save_dir = cfg.save_dir + '/finetune_1batch_test'
if cfg.is_baseline:
    save_dir += '_baseline'
else:
    save_dir += '_CL_' + cfg.start_str

save_dir += '/' + cfg.datatype + '_' + cfg.no_of_tr_imgs + '_' + cfg.loss_function
if cfg.data_aug:
    save_dir += '_data_aug'
save_dir += '/tr_comb_' + str(cfg.run_num) + '_LR_ft_' + str(cfg.lr_finetune) + '/'

print('Creating', save_dir)
pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)

modelSavePath = save_dir + 'weights_{epoch:02d}.hdf5'
csvPath = save_dir + 'training.log'
callbacks = get_callbacks(csvPath)

initial_epoch = 0
num_epochs = cfg.num_epochs

history = ae_reg.fit(
    x_tr, y_tr,                     # <<-- always the same batch
    epochs=num_epochs,
    batch_size=x_tr.shape[0],       # one full batch
    verbose=1,
    callbacks=callbacks,
    validation_data=(x_tr, y_tr), # <<-- always same val batch
    initial_epoch=initial_epoch,
    shuffle=False                   # no shuffling between epochs
)
ae_reg.save_weights(
    save_dir + 'weights_' + str(num_epochs) + '_Final.hdf5',
    overwrite=True,
    save_format='h5',
    options=None
)

cfg_txt_name = save_dir + 'config_params.txt'
with open(cfg_txt_name, 'w') as f:
    for name, value in cfg.__dict__.items():
        f.write('{} = {!r}\n'.format(name, value))



# y_pred_tr = ae_reg.predict(x_tr)
# y_pred_val = ae_reg.predict(x_val)

# import numpy as np

# np.savez_compressed(
#     "sanity_batch_outputs.npz",
#     x_tr=x_tr,
#     y_tr=y_tr,
#     y_pred_tr=y_pred_tr,
#     x_val=x_val,
#     y_val=y_val,
#     y_pred_val=y_pred_val,
# )

# print("Saved sanity_batch_outputs.npz")
