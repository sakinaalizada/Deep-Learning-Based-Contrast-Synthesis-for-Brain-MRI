import sys
sys.path.append("/gpfs/scratch/sa9063/sakina-thesis/CCL-Synthetis/CCL-Synthetis")
from Synthesis import synth_config as cfg
from utils.model_utils import modelObj
import numpy as np

# 1) Build the same baseline model
model = modelObj(cfg).synth_unet()
model.summary()

#%%












wpath = "/gpfs/scratch/sa9063/sakina-thesis/outputs/nyu_mets_t1-t2-flair-50loss1/rand_deform/finetune_Hybride_baseline/t1-t2-flair-50loss1_tr1_Hybride/tr_comb_Exp1_LR_ft_0.0001/weights_40Final.hdf5"  # <-- put full path to weights_40Final.hdf5
print("Loading:", wpath)
model.load_weights(wpath)

# 3) Check for NaNs in weights
has_nan = False
for w in model.weights:
    if np.isnan(w.numpy()).any():
        print("NaNs found in weight:", w.name)
        has_nan = True
        break

if not has_nan:
    print("✅ No NaNs in any weights")