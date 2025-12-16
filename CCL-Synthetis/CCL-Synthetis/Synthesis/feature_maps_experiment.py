import os, sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import h5py

# make sure repo is on path
sys.path.append("/gpfs/scratch/sa9063/sakina-thesis/CCL-Synthetis/CCL-Synthetis/")

from Synthesis import synth_config as cfg
from utils.model_utils import modelObj

# ---- 1) Force CL mode + correct weights ----
cfg.is_baseline = 0
cfg.is_CL = 1

# make sure CL_param_wts points to your RADS checkpoint
# (already set in synth_config)
print("Using CL weights:", cfg.CL_param_wts)

# ---- 2) Build model + load weights ----
model = modelObj(cfg)
ae_reg = model.synth_unet()
ae_reg.load_weights(cfg.CL_param_wts, by_name=True)
print("Loaded weights.")

# ---- 3) Build feature-model for penultimate layer (seg_op2) ----
# last Activation layer in the model is seg_op2
act_layers = [l for l in ae_reg.layers if isinstance(l, tf.keras.layers.Activation)]
print("Activation layers:", [l.name for l in act_layers])
penultimate_layer = act_layers[-1]   # seg_op2

feature_model = tf.keras.Model(
    inputs=ae_reg.input,
    outputs=penultimate_layer.output
)
print("Feature model output shape:", feature_model.output_shape)

# ---- 4) Load a few NYU Mets slices ----
with h5py.File(cfg.hdf5_train_img_filename, "r") as f:
    img = f["img"]   # shape (N, 160, 160, 4)
    print("HDF5 img shape:", img.shape)

    # pick a few random indices
    idxs = np.random.choice(img.shape[0], size=1, replace=False)

    os.makedirs("feature_maps_experiment", exist_ok=True)

    for k, idx in enumerate(idxs):
        x = img[idx:idx+1, ..., cfg.contrast_idx]   # (1,160,160,3)
        feats = feature_model.predict(x)[0]         # (160,160,C)

        # save first 8 channels
        for c in range(min(8, feats.shape[-1])):
            plt.imshow(feats[..., c], cmap="gray")
            plt.axis("off")
            out_path = f"feature_maps_experiment/sample{idx}_chan{c}.png"
            plt.savefig(out_path, bbox_inches="tight")
            plt.close()
            print("Saved", out_path)