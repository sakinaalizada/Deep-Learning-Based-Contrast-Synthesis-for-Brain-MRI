import sys
sys.path.append("/gpfs/scratch/sa9063/sakina-thesis/CCL-Synthetis/CCL-Synthetis")


import h5py
import numpy as np

from Datagen.h5_pretrain_Synth_Data_Generator import DataLoaderObj
from Synthesis import synth_config as cfg
import numpy as np
from utils.model_utils import modelObj


hdf5_path = '/gpfs/data/fenglab/ParisimaAbdali/pa2297/Training/t1-t2-flair/Datah5/Zmean350-synth/train_t1-t2-flair_pretrain_img_zmean.hdf5'


print("\n=== HDF5 SANITY CHECK ===\n")

with h5py.File(hdf5_path, "r") as f:
    keys = list(f.keys())
    print("Datasets found:", keys)

    dname = keys[0]
    dset = f[dname]

    print(f"\nDataset '{dname}' info:")
    print("Shape:", dset.shape)
    print("Dtype:", dset.dtype)

    sample = dset[0]
    print("\nOne sample shape:", sample.shape)

    print("\nStats for sample:")
    print("  min:", np.min(sample))
    print("  max:", np.max(sample))
    print("  mean:", np.mean(sample))

    print("\nContains NaN?", np.isnan(sample).any())

    if sample.ndim == 3:
        print("Number of channels:", sample.shape[-1])
    else:
        print("Warning: sample is not (H,W,C)!")

print("\n=== DONE ===\n")

train_gen = DataLoaderObj(cfg, train_flag=True)

x_tr, y_tr = train_gen[0]
print("x_tr shape:", x_tr.shape)
print("y_tr shape:", y_tr.shape)