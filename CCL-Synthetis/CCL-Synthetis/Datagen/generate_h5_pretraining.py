"""
@author: lavanya
Provides example script to generate HDF5 files for pretraining  with the MR-contrast guided contrastive learning approach
Script combines nii data with the constraint maps generated using generate_constraint_maps.py
This section can be replaced by any script that generates training h5 files with
'img' as N x Height x Width x Channels (float64)
'param' as N x Height x Width x 1       (int64)
"""

import nibabel as nib
import h5py  
import numpy as np
import os, natsort

import sys
import scipy.io as sio
import csv

sys.path.append('/gpfs/home/sa9063/sakina-thesis/CCL-Synthetis/CCL-Synthetis')

from utils.utils import myCrop3D
from utils.utils import contrastStretch
from utils.utils import normalize_img, normalize_img_zmean


datadir = '/gpfs/data/fenglab/LiFeng/NYUMets/NYUMets/data/imaging/patientId/'    # location of pre-processed nii files for pretraining
# labeldir           = '/gpfs/scratch/pa2297/CM/t1-t2-flair/Constraint_Maps/'  # location of constraint maps for pretraining
# save_dir           = '/mets_dataset/' + 'results/'  
save_dir = '/gpfs/scratch/sa9063/sakina-thesis/Results/'
datatype           = 'test'    # options are train/val
contrast_type      = 't1-t2-flair'
opShape            = (160,160)
zmean_norm         = True   # perform zero mean unit std normalization, suited for multi-parametric multi-contrast data
# num_param_cluster  = '20'



import pandas as pd

meta_csv = "/gpfs/data/fenglab/LiFeng/NYUMets/NYUMets/metadata/20220825_imaging.csv"
df = pd.read_csv(meta_csv)

df4 = df[(df.T1 == 1) & (df.CT1 == 1) & (df.T2 == 1) & (df.FLAIR == 1)]

four_contrast_ids = [
    f"{p}/{s}" for p, s in zip(df4.PatientID, df4.StudyID)
]


''' Use an appropriate dataloader to load multi-contrast training data
    Example here is for the brain tumor segmentation dataset'''

def load_unl_brats_img(datadir, subName, opShape): 
    ''' Loads a 4D volume from the brats dataset HxWxDxT where the contrasts are [T1Gd, T2w, T1w, T2-FLAIR]'''
    print('Loading MP-MR images for ', subName)
    data_suffix = ['CT1.nii', 'T2.nii', 'T1.nii', 'FLAIR.nii']
    patient, study = subName.split("/")
    sub_img = []
    for suffix in data_suffix:
        temp = nib.load(f"{datadir}{patient}/studyId/{study}/{suffix}")
        temp_data = temp.get_fdata()
        if temp_data.size == 0 or np.all(temp_data == 0):
            raise ValueError("Empty or zero image")
        temp = np.rot90(temp_data, -1)
        temp = myCrop3D(temp, opShape)
        # generate a brain mask from the first volume. If mask is available, skip this step
        if suffix == data_suffix[0]:  
            mask = np.zeros(temp.shape)
            mask[temp > 0] = 1
        # histogram based channel-wise contrast stretching
        temp = contrastStretch(temp, mask, 0.01, 99.9)
        if zmean_norm:
            temp = normalize_img_zmean(temp, mask)
        else:
            temp = normalize_img(temp)
        sub_img.append(temp)
    sub_img = np.stack((sub_img), axis=-1)
    return  sub_img 

# def load_brats_cluster(subName, opShape): 
#     print('Loading constraint maps for ', subName)
#     temp = sio.loadmat(labeldir + subName + '/' + 'Constraint_map_'+ num_param_cluster +'.mat')
#     print('Loading successful')
#     param_label = temp['param']
#     param_label = myCrop3D(param_label, opShape)
#     return  param_label 
 
#%%

unique_patients = df4["PatientID"].unique()

np.random.seed(25000)
np.random.shuffle(unique_patients)

num_patients = len(unique_patients)

train_frac = 0.8
val_frac   = 0.1

num_train_pat = int(train_frac * num_patients)
num_val_pat   = int(val_frac * num_patients)

train_patients = set(unique_patients[:num_train_pat])
val_patients   = set(unique_patients[num_train_pat:num_train_pat + num_val_pat])
test_patients  = set(unique_patients[num_train_pat + num_val_pat:])

df4["sub_id"] = df4["PatientID"].astype(str) + "/" + df4["StudyID"].astype(str)

train_subs = df4[df4["PatientID"].isin(train_patients)]["sub_id"].tolist()
val_subs   = df4[df4["PatientID"].isin(val_patients)]["sub_id"].tolist()
test_subs  = df4[df4["PatientID"].isin(test_patients)]["sub_id"].tolist()

if datatype == "train":
    wd_trunc = train_subs
elif datatype == "val":
    wd_trunc = val_subs
elif datatype == "test":
    wd_trunc = test_subs

# sub_list = four_contrast_ids
# np.random.seed(seed=25000)
# np.random.shuffle(sub_list)
# '''Edited by Parisima'''
# num_train_files = 300  # Number of files for training
# num_val_files = 150   # Number of files for validation

''' Select an appropriate split of training and validation data for pretraining'''
# if datatype == 'train':
#     wd_trunc = sub_list[:-20]
# elif datatype == 'val':
#     wd_trunc = sub_list[-20:]
# total = len(sub_list)

# num_train = int(0.8 * total)   # 80% training
# num_val   = int(0.1 * total)   # 10% validation
# remaining automatically becomes test set

# if datatype == 'train':
#     wd_trunc = sub_list[:num_train]

# elif datatype == 'val':
#     wd_trunc = sub_list[num_train:num_train + num_val]

# elif datatype == 'test':
#     wd_trunc = sub_list[num_train + num_val:]

if zmean_norm:
    imgs_h5py_filename = save_dir + datatype + '_' + contrast_type + '_pretrain_img_zmean.hdf5' 
else:
    imgs_h5py_filename = save_dir + datatype + '_' + contrast_type + '_pretrain_img.hdf5' 
     
constraintmap_h5py_filename = save_dir + datatype + '_' + contrast_type + '_pretrain_constraint_map.hdf5'  

#%% Generate training and validation files with NxHxWxT for image and HxWxDx1 for constraint maps
''' 
The following section process 30 subjects at a time.
This section can be replaced by any script that generates training h5 files with
'img' as N x Height x Width x Channels (float64)
'param' as N x Height x Width x 1       (int64)
'''

ctr = 0
init_Flag = True
imgs = []
num_vols = 10  # process 30 subjects at a time


for subName in wd_trunc:
    print('SubName', subName, ctr)

    # Load images only
    try:
        sub_img = load_unl_brats_img(datadir, subName, opShape)
    except Exception as e:
        print(f"Skipping {subName} — error: {e}")
        continue

    # transpose and crop
    sub_img = np.transpose(sub_img, (2,0,1,3))
    sub_img = sub_img[40:120]

    # ensure Z dimension is correct (should be 80)
    if sub_img.shape[0] != 80:
        print(f"Skipping {subName} — incorrect Z dimension: {sub_img.shape[0]}")
        continue

    imgs.append(sub_img)

    img_z, img_x, img_y, num_channels = sub_img.shape
    ctr += 1

    # write full chunks of 30 subjects
    if ctr == num_vols:
        print('Writing chunk to hdf5')

        imgs = np.stack(imgs)  # shape (30, z, x, y, c)
        chunk = imgs.reshape(-1, img_x, img_y, num_channels)

        if init_Flag:
            print('Writing new file')
            with h5py.File(imgs_h5py_filename, 'w') as f:
                dset = f.create_dataset(
                    "img",
                    chunk.shape,
                    maxshape=(None, img_x, img_y, num_channels),
                    chunks=True,
                    dtype='float64'
                )
                dset[:] = chunk
            init_Flag = False

        else:
            print('Appending')
            with h5py.File(imgs_h5py_filename, 'a') as f:
                dset = f["img"]
                old = dset.shape[0]
                dset.resize(old + chunk.shape[0], axis=0)
                dset[-chunk.shape[0]:] = chunk

        ctr = 0
        imgs = []  # reset list

# ---- Write leftover subjects (fewer than 30) ----
if len(imgs) > 0:
    print("Writing leftover subjects")

    imgs = np.stack(imgs)  # shape (N, z, x, y, c)
    chunk = imgs.reshape(-1, img_x, img_y, num_channels)

    if init_Flag:
        print('Writing new file (no chunk written before)')
        with h5py.File(imgs_h5py_filename, 'w') as f:
            dset = f.create_dataset(
                "img",
                chunk.shape,
                maxshape=(None, img_x, img_y, num_channels),
                chunks=True,
                dtype='float64'
            )
            dset[:] = chunk
    else:
        print("Appending leftover")
        with h5py.File(imgs_h5py_filename, 'a') as f:
            dset = f["img"]
            old = dset.shape[0]
            dset.resize(old + chunk.shape[0], axis=0)
            dset[-chunk.shape[0]:] = chunk