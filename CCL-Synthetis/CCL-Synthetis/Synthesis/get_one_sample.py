import h5py
import numpy as np
import nibabel as nib
import os

# -------------------------
# Paths
# -------------------------
train_h5 = "/gpfs/scratch/sa9063/sakina-thesis/Results/train_t1-t2-flair_pretrain_img_zmean.hdf5"
val_h5   = "/gpfs/scratch/sa9063/sakina-thesis/Results/val_t1-t2-flair_pretrain_img_zmean.hdf5"
test_h5  = "/gpfs/scratch/sa9063/sakina-thesis/Results/test_t1-t2-flair_pretrain_img_zmean.hdf5"

out_dir = "./hdf5_samples_as_nii"
os.makedirs(out_dir, exist_ok=True)

IMG_KEY = "img"

# -------------------------
# Helper: save 4 contrasts
# -------------------------
def save_sample_as_nii(h5_path, split_name, idx=0):
    with h5py.File(h5_path, "r") as f:
        x = f[IMG_KEY][idx]   # (H, W, 4)

    assert x.shape[-1] == 4, "Expected 4 contrasts"

    contrast_names = ["CT1", "T2", "T1", "FLAIR"]

    for c, name in enumerate(contrast_names):
        img = x[..., c]

        nii = nib.Nifti1Image(img.astype(np.float32), affine=np.eye(4))
        out_path = os.path.join(out_dir, f"{split_name}_{name}.nii.gz")
        nib.save(nii, out_path)

    print(f"Saved {split_name} sample idx={idx}")

# -------------------------
# Extract ONE sample each
# -------------------------
save_sample_as_nii(train_h5, "train", idx=0)
save_sample_as_nii(val_h5,   "val",   idx=0)
save_sample_as_nii(test_h5,  "test",  idx=0)
