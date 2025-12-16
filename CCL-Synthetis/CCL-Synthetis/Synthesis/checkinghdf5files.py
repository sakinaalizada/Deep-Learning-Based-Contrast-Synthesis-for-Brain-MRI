import sys
sys.path.append("/gpfs/scratch/sa9063/sakina-thesis/CCL-Synthetis/CCL-Synthetis")

from Datagen.h5_pretrain_Synth_Data_Generator import DataLoaderObj
from Synthesis import synth_config as cfg
import numpy as np
from utils.model_utils import modelObj

gen = DataLoaderObj(cfg, train_flag=True)
x, y = gen[0]

print("x shape:", x.shape)
print("y shape:", y.shape)
print("x min/max:", np.min(x), np.max(x))
print("y min/max:", np.min(y), np.max(y))

# import h5py
# import numpy as np

# path = '/gpfs/scratch/sa9063/sakina-thesis/Results/test_t1-t2-flair_pretrain_img_zmean.hdf5'

# with h5py.File(path, 'r') as f:
#     x = f['img']            # shape: (319040, 160, 160, 4)
#     n = x.shape[0]
#     chunk = 64              # number of samples to load at a time

#     global_min = np.inf
#     global_max = -np.inf
#     has_nan = False
#     has_inf = False

#     for start in range(0, n, chunk):
#         end = min(start + chunk, n)
#         xb = x[start:end]   # only [chunk] samples in memory

#         # update global stats
#         global_min = min(global_min, xb.min())
#         global_max = max(global_max, xb.max())
#         has_nan = has_nan or np.isnan(xb).any()
#         has_inf = has_inf or np.isinf(xb).any()

#         print(f"Batch {start:6d}-{end:6d}: "
#               f"min={xb.min():.3g}, max={xb.max():.3g}, "
#               f"NaN={np.isnan(xb).any()}, Inf={np.isinf(xb).any()}")

#         # optional: break early if you already see problems
#         # if has_nan or has_inf:
#         #     break

#     print("==== GLOBAL STATS ====")
#     print("min:", global_min, "max:", global_max,
#           "any NaN:", has_nan, "any Inf:", has_inf)




# # import h5py, numpy as np

# # path = '/gpfs/scratch/sa9063/sakina-thesis/Results/train_t1-t2-flair_pretrain_img_zmean.hdf5'

# # with h5py.File(path, 'r+') as f:
# #     dset = f['img']
# #     n = dset.shape[0]
# #     chunk = 64
# #     for start in range(0, n, chunk):
# #         end = min(start + chunk, n)
# #         xb = dset[start:end]
# #         if np.isnan(xb).any() or np.isinf(xb).any():
# #             xb = np.nan_to_num(xb, nan=0.0, posinf=0.0, neginf=0.0)
# #             dset[start:end] = xb
# #             print(f"Cleaned NaNs/Infs in [{start}, {end})")

