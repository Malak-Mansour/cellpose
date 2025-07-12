from cellpose import io, models, core, train
from skimage.measure import label
from pathlib import Path
import numpy as np
import re

# IMAGE & MASK FOLDERS
image_dir = Path("../organized_masks_data_7_7_2025/Jurkat/Dynamic/5. Jurkat_ON_OFF_10V_R1/10v_20s-40s_nd_089")
mask_dir = Path("../organized_masks_data_7_7_2025/Jurkat/Dynamic/5. Jurkat_ON_OFF_10V_R1/Modified_masks")

# GLOB FILES
image_files = sorted(image_dir.glob("t*.tif"))
mask_files = sorted(mask_dir.glob("masks*.tif"))

# MATCHING BASED ON INDEX NUMBER (e.g., t01 <-> masks01)
def extract_index(filename):
    match = re.search(r"(\d+)", filename.stem)
    return int(match.group(1)) if match else None

image_map = {extract_index(f): f for f in image_files}
mask_map = {extract_index(f): f for f in mask_files}

# Ensure both have same indices
common_indices = sorted(set(image_map.keys()) & set(mask_map.keys()))
print(f"✅ Found {len(common_indices)} matched image-mask pairs.")


def bin_to_labels(mask):
    """Convert 16-bit grayscale binary mask to labeled instance mask"""
    threshold = np.max(mask) * 0.5
    binary = mask > threshold
    labeled = label(binary.astype(np.uint8), connectivity=1)
    return labeled


if len(common_indices) == 0:
    raise RuntimeError("❌ No matching image/mask pairs found.")

print(f"✅ Found {len(common_indices)} matched image/mask pairs")

# Load as arrays
image_arrays = [io.imread(image_map[i]) for i in common_indices]
mask_arrays = [bin_to_labels(io.imread(mask_map[i])) for i in common_indices]

print(f"✅ Loaded {len(image_arrays)} image/mask pairs.")

# GPU check
io.logger_setup()
if not core.use_gpu():
    raise RuntimeError("No GPU found")

# Initialize base model
model = models.CellposeModel(gpu=True)

# Train
print("🚀 Starting training...")
new_model_path, train_losses, test_losses = train.train_seg(
    model.net,
    train_data=image_arrays,
    train_labels=mask_arrays,
    # test_data=[],
    # test_labels=[],
    batch_size=1,
    n_epochs=100,
    learning_rate=1e-5,
    weight_decay=0.1,
    nimg_per_epoch=max(2, len(image_arrays)),
    model_name="jurkat_mismatched_name_model"
)

print(f"✅ Training complete. Model saved to: {new_model_path}")

# image_paths = [str(image_map[i]) for i in common_indices]
# mask_paths = [str(mask_map[i]) for i in common_indices]

# # GPU check
# from cellpose import io
# io.logger_setup()
# if not core.use_gpu():
#     raise RuntimeError("No GPU found. Please enable GPU runtime.")

# # INIT MODEL
# model = models.CellposeModel(gpu=True)

# # TRAIN THE MODEL
# model_name = "jurkat_model"

# new_model_path, train_losses, test_losses = train.train_seg(
#     model.net,
#     train_data=image_paths,
#     train_labels=mask_paths,
#     test_data=[],            # optional
#     test_labels=[],
#     batch_size=1,
#     n_epochs=100,
#     learning_rate=1e-5,
#     weight_decay=0.1,
#     nimg_per_epoch=max(2, len(image_paths)),
#     model_name=model_name
# )
