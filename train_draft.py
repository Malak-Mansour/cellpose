# from cellpose import io, models, core, train
# from skimage.measure import label
# from pathlib import Path
# import numpy as np
# import re

# # IMAGE & MASK FOLDERS
# image_dir = Path("../organized_masks_data_7_7_2025/Jurkat/Dynamic/5. Jurkat_ON_OFF_10V_R1/10v_20s-40s_nd_089")
# mask_dir = Path("../organized_masks_data_7_7_2025/Jurkat/Dynamic/5. Jurkat_ON_OFF_10V_R1/Modified_masks")

# # GLOB FILES
# image_files = sorted(image_dir.glob("t*.tif"))
# mask_files = sorted(mask_dir.glob("masks*.tif"))

# # MATCHING BASED ON INDEX NUMBER (e.g., t01 <-> masks01)
# def extract_index(filename):
#     match = re.search(r"(\d+)", filename.stem)
#     return int(match.group(1)) if match else None

# image_map = {extract_index(f): f for f in image_files}
# mask_map = {extract_index(f): f for f in mask_files}

# # Ensure both have same indices
# common_indices = sorted(set(image_map.keys()) & set(mask_map.keys()))
# print(f"✅ Found {len(common_indices)} matched image-mask pairs.")


# def bin_to_labels(mask):
#     """Convert 16-bit grayscale binary mask to labeled instance mask"""
#     threshold = np.max(mask) * 0.5
#     binary = mask > threshold
#     labeled = label(binary.astype(np.uint8), connectivity=1)
#     return labeled


# if len(common_indices) == 0:
#     raise RuntimeError("❌ No matching image/mask pairs found.")

# print(f"✅ Found {len(common_indices)} matched image/mask pairs")

# # Load as arrays
# image_arrays = [io.imread(image_map[i]) for i in common_indices]
# mask_arrays = [bin_to_labels(io.imread(mask_map[i])) for i in common_indices]

# print(f"✅ Loaded {len(image_arrays)} image/mask pairs.")

# # GPU check
# io.logger_setup()
# if not core.use_gpu():
#     raise RuntimeError("No GPU found")

# # Initialize base model
# model = models.CellposeModel(gpu=True)

# # Train
# print("🚀 Starting training...")
# new_model_path, train_losses, test_losses = train.train_seg(
#     model.net,
#     train_data=image_arrays,
#     train_labels=mask_arrays,
#     # test_data=[],
#     # test_labels=[],
#     batch_size=1,
#     n_epochs=100,
#     learning_rate=1e-5,
#     weight_decay=0.1,
#     nimg_per_epoch=max(2, len(image_arrays)),
#     model_name="jurkat_mismatched_name_model"
# )

# print(f"✅ Training complete. Model saved to: {new_model_path}")











# from cellpose import io, models, core, train
# from skimage.measure import label
# from pathlib import Path
# import numpy as np
# import re

# # ------------------ Helper Functions ------------------ #

# def extract_index(filename):
#     """Extracts the numeric index from a filename like 't01.tif' or 'masks01.tif'"""
#     match = re.search(r"(\d+)", filename.stem)
#     return int(match.group(1)) if match else None

# def bin_to_labels(mask):
#     """Convert 16-bit grayscale binary mask to labeled instance mask"""
#     threshold = np.max(mask) * 0.5
#     binary = mask > threshold
#     labeled = label(binary.astype(np.uint8), connectivity=1)
#     return labeled

# def load_image_mask_pairs(image_dir, mask_dir):
#     image_files = sorted(image_dir.glob("t*.tif"))
#     mask_files = sorted(mask_dir.glob("masks*.tif"))

#     image_map = {extract_index(f): f for f in image_files}
#     mask_map = {extract_index(f): f for f in mask_files}

#     common_indices = sorted(set(image_map.keys()) & set(mask_map.keys()))
#     print(f"📂 {image_dir.name} → {len(common_indices)} matched pairs")

#     images = [io.imread(image_map[i]) for i in common_indices]
#     masks = [bin_to_labels(io.imread(mask_map[i])) for i in common_indices]
#     print(masks)
#     return images, masks

# # ------------------ Paths ------------------ #

# # List of image/mask folder pairs
# folder_pairs = [
#     ("../organized_masks_data_7_7_2025/Jurkat/Dynamic/5. Jurkat_ON_OFF_10V_R1/10v_20s-40s_nd_089",
#     "../organized_masks_data_7_7_2025/Jurkat/Dynamic/5. Jurkat_ON_OFF_10V_R1/Modified_masks"),

#     ("../organized_masks_data_7_7_2025/Jurkat/Dynamic/5. Jurkat_ON_OFF_10V_R3/10v_20s-40s_nd_091",
#      "../organized_masks_data_7_7_2025/Jurkat/Dynamic/5. Jurkat_ON_OFF_10V_R3/Modified_masks"),
    
#     ("../organized_masks_data_7_7_2025/SKBR3/Dynamic/6. SKBR3_ON_OFF_10V_R1/L6",
#      "../organized_masks_data_7_7_2025/SKBR3/Dynamic/6. SKBR3_ON_OFF_10V_R1/Modified_masks"),
    
#     ("../organized_masks_data_7_7_2025/SKBR3/Dynamic/6. SKBR3_ON_OFF_10V_R2/L4",
#      "../organized_masks_data_7_7_2025/SKBR3/Dynamic/6. SKBR3_ON_OFF_10V_R2/Modified_masks"),
    
#     ("../organized_masks_data_7_7_2025/SKBR3/Dynamic/6. SKBR3_ON_OFF_10V_R3/L3_001",
#      "../organized_masks_data_7_7_2025/SKBR3/Dynamic/6. SKBR3_ON_OFF_10V_R3/Modified_masks"),
    
#     ("../organized_masks_data_7_7_2025/SKBR3/Static/3. SKBR3_Static_R1/L3",
#      "../organized_masks_data_7_7_2025/SKBR3/Static/3. SKBR3_Static_R1/Modified_masks"),
    
#     ("../organized_masks_data_7_7_2025/SKBR3/Static/3. SKBR3_Static_R2/L5",
#      "../organized_masks_data_7_7_2025/SKBR3/Static/3. SKBR3_Static_R2/Modified_masks"),
    
#     ("../organized_masks_data_7_7_2025/SKBR3/Static/3. SKBR3_Static_R3/L2",
#      "../organized_masks_data_7_7_2025/SKBR3/Static/3. SKBR3_Static_R3/Modified_masks"),
# ]

# # ------------------ Load Data ------------------ #
# # Split into mini-batches (e.g., 3 folders at a time)
# batch_size = 1
# batches = [folder_pairs[i:i+batch_size] for i in range(0, len(folder_pairs), batch_size)]

# model_path = None  # 🔧 Initialize here

# for stage_idx, folder_batch in enumerate(batches):
#     print(f"\n🧪 Training batch {stage_idx+1}/{len(batches)}")

#     all_images, all_masks = [], []

#     for img_path, mask_path in folder_batch:
#         images, masks = load_image_mask_pairs(Path(img_path), Path(mask_path))
#         all_images.extend(images)
#         all_masks.extend(masks)

#     print(f"✅ Batch {stage_idx+1}: {len(all_images)} images, {len(all_masks)} masks")

#     if model_path:
#         model = models.CellposeModel(gpu=True, pretrained_model=model_path)
#     else:
#         model = models.CellposeModel(gpu=True)  # Start with default

#     model_name = f"fine_model_batch_{stage_idx+1}"
#     print(f"🚀 Training model: {model_name}")

#     model_path, train_losses, test_losses = train.train_seg(
#         model.net,
#         train_data=all_images,
#         train_labels=all_masks,
#         batch_size=1,
#         n_epochs=100,
#         learning_rate=1e-5,
#         weight_decay=0.1,
#         nimg_per_epoch=min(60, len(all_images)),
#         model_name=model_name
#     )

#     print(f"💾 Saved model from batch {stage_idx+1} to: {model_path}")

# print(f"\n✅ All batches complete. Final model saved at: {model_path}")
















# from cellpose import io, models, core, train
# from skimage.measure import label
# from pathlib import Path
# import numpy as np
# import re
# import logging
# from logging.handlers import RotatingFileHandler

# # ------------------ Helper Functions ------------------ #

# def extract_index(filename):
#     match = re.search(r"(\d+)", filename.stem)
#     return int(match.group(1)) if match else None

# def bin_to_labels(mask):
#     threshold = np.max(mask) * 0.5
#     binary = mask > threshold
#     labeled = label(binary.astype(np.uint8), connectivity=1)
#     return labeled

# def load_image_mask_pairs(image_dir, mask_dir):
#     image_files = sorted(image_dir.glob("t*.tif"))
#     mask_files = sorted(mask_dir.glob("masks*.tif"))

#     image_map = {extract_index(f): f for f in image_files}
#     mask_map = {extract_index(f): f for f in mask_files}

#     common_indices = sorted(set(image_map.keys()) & set(mask_map.keys()))
#     print(f"📂 {image_dir.name} → {len(common_indices)} matched pairs")

#     images = [io.imread(image_map[i]) for i in common_indices]
#     masks = [bin_to_labels(io.imread(mask_map[i])) for i in common_indices]
#     return images, masks

# def train_on_folder(image_dir, mask_dir, model_path_in=None, model_name_out="finetuned_model"):
#     images, masks = load_image_mask_pairs(image_dir, mask_dir)

#     if len(images) == 0:
#         print(f"⚠️ Skipping {image_dir.name} — no matching pairs found.")
#         return model_path_in
    


    
#      # -------- Robust Logging Setup --------
#     log_file = f"/l/users/malak.mansour/DEP/cellpose/logs/train_stage_{idx}.log"
#     log_dir = Path(log_file).parent
#     log_dir.mkdir(parents=True, exist_ok=True)

#     # Clear existing handlers and set up a rotating file handler
#     for handler in logging.root.handlers[:]:
#         logging.root.removeHandler(handler)

#     handler = RotatingFileHandler(log_file, maxBytes=5 * 256 * 256, backupCount=3)
#     formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
#     handler.setFormatter(formatter)

#     logger = logging.getLogger()
#     logger.setLevel(logging.INFO)
#     logger.addHandler(handler)

#     # Optional: also log to console
#     console = logging.StreamHandler()
#     console.setFormatter(formatter)
#     logger.addHandler(console)

#     logger.info(f"🚀 Starting training on folder: {image_dir.name}")




#     # Setup logging and GPU
#     io.logger_setup()
#     if not core.use_gpu():
#         raise RuntimeError("❌ No GPU found")

#     if model_path_in:
#         print(f"📥 Loading model from: {model_path_in}")
#         model = models.CellposeModel(gpu=True, pretrained_model=Path(model_path_in).expanduser())
#     else:
#         print("📥 Using default pretrained Cellpose model")
#         model = models.CellposeModel(gpu=True)

#     print(f"🚀 Training on: {image_dir.name}")
#     new_model_path, _, _ = train.train_seg(
#         model.net,
#         train_data=images,
#         train_labels=masks,
#         batch_size=1,
#         n_epochs=100,
#         learning_rate=1e-5,
#         weight_decay=0.1,
#         nimg_per_epoch=max(2, len(images)),
#         model_name=model_name_out
#     )
#     print(f"✅ Saved model to: {new_model_path}")
#     logger.info(f"✅ Saved model to: {new_model_path}")
#     return new_model_path


# # ------------------ Dataset List (train sequentially) ------------------ #

# folder_pairs = [
#     # ("../organized_masks_data_7_7_2025/Jurkat/Dynamic/5. Jurkat_ON_OFF_10V_R1/10v_20s-40s_nd_089",
#     #  "../organized_masks_data_7_7_2025/Jurkat/Dynamic/5. Jurkat_ON_OFF_10V_R1/Modified_masks"),

#     # ("../organized_masks_data_7_7_2025/Jurkat/Dynamic/5. Jurkat_ON_OFF_10V_R3/10v_20s-40s_nd_091",
#     #  "../organized_masks_data_7_7_2025/Jurkat/Dynamic/5. Jurkat_ON_OFF_10V_R3/Modified_masks"),

#     # ("../organized_masks_data_7_7_2025/SKBR3/Dynamic/6. SKBR3_ON_OFF_10V_R1/L6",
#     #  "../organized_masks_data_7_7_2025/SKBR3/Dynamic/6. SKBR3_ON_OFF_10V_R1/Modified_masks"),

#     # ("../organized_masks_data_7_7_2025/SKBR3/Dynamic/6. SKBR3_ON_OFF_10V_R2/L4",
#     #  "../organized_masks_data_7_7_2025/SKBR3/Dynamic/6. SKBR3_ON_OFF_10V_R2/Modified_masks"),
    
#     ("../organized_masks_data_7_7_2025/SKBR3/Dynamic/6. SKBR3_ON_OFF_10V_R3/L3_001",
#      "../organized_masks_data_7_7_2025/SKBR3/Dynamic/6. SKBR3_ON_OFF_10V_R3/Modified_masks"),
    
#     ("../organized_masks_data_7_7_2025/SKBR3/Static/3. SKBR3_Static_R1/L3",
#      "../organized_masks_data_7_7_2025/SKBR3/Static/3. SKBR3_Static_R1/Modified_masks"),
    
#     ("../organized_masks_data_7_7_2025/SKBR3/Static/3. SKBR3_Static_R2/L5",
#      "../organized_masks_data_7_7_2025/SKBR3/Static/3. SKBR3_Static_R2/Modified_masks"),
    
#     ("../organized_masks_data_7_7_2025/SKBR3/Static/3. SKBR3_Static_R3/L2",
#      "../organized_masks_data_7_7_2025/SKBR3/Static/3. SKBR3_Static_R3/Modified_masks"),
# ]

# # ------------------ Run Sequential Training ------------------ #

# # ------------------ Start from fine_model_stage_4 ------------------ #



# if __name__ == "__main__":
#     # Resume from fine_model_stage_4
#     current_model_path = Path("~/.cellpose/models/fine_model_stage_4").expanduser()

#     # Train on the 4 remaining folders
#     for idx, (img_dir, mask_dir) in enumerate(folder_pairs, start=5):
#         model_name_out = f"fine_model_stage_{idx}"
#         current_model_path = train_on_folder(
#             Path(img_dir),
#             Path(mask_dir),
#             model_path_in=current_model_path,
#             model_name_out=model_name_out
#         )
















# from cellpose import io, models, core, train
# from skimage.measure import label
# from pathlib import Path
# import numpy as np
# import re
# import matplotlib.pyplot as plt
# from skimage.transform import resize

# # ------------------ Helper Functions ------------------ #

# def extract_index(filename):
#     match = re.search(r"(\d+)", filename.stem)
#     return int(match.group(1)) if match else None

# def bin_to_labels(mask):
#     threshold = np.max(mask) * 0.5
#     binary = mask > threshold
#     labeled = label(binary.astype(np.uint8), connectivity=1)
#     return labeled

# def sample_true_negatives_from_resized_mask(resized_mask, num_samples=30):
#     tn_coords = []
#     bg_coords = np.argwhere(resized_mask == 0)
#     if len(bg_coords) > 0:
#         sampled = bg_coords[np.random.choice(len(bg_coords), min(num_samples, len(bg_coords)), replace=False)]
#         for yx in sampled:
#             tn_coords.append((yx[0], yx[1]))
#     return tn_coords

# def load_image_mask_pairs(image_dir, mask_dir, target_shape=(256, 256), num_tns=30):
#     image_files = sorted(image_dir.glob("t*.tif"))
#     mask_files = sorted(mask_dir.glob("masks*.tif"))

#     image_map = {extract_index(f): f for f in image_files}
#     mask_map = {extract_index(f): f for f in mask_files}

#     common_indices = sorted(set(image_map.keys()) & set(mask_map.keys()))
#     print(f"📂 {image_dir.name} → {len(common_indices)} matched pairs")

#     images, masks, tns_all = [], [], []

#     for i in common_indices:
#         img = io.imread(image_map[i])
#         mask = bin_to_labels(io.imread(mask_map[i]))

#         resized_mask = resize(mask.astype(np.uint8), target_shape, order=0, preserve_range=True, anti_aliasing=False).astype(np.uint8)
#         tn_coords = sample_true_negatives_from_resized_mask(resized_mask, num_tns)

#         images.append(img)
#         masks.append(mask)
#         tns_all.append(tn_coords)

#     return images, masks, tns_all

# # ------------------ Paths ------------------ #

# folder_pairs = [
#     ("../organized_masks_data_7_7_2025/Jurkat/Dynamic/5. Jurkat_ON_OFF_10V_R1/10v_20s-40s_nd_089",
#      "../organized_masks_data_7_7_2025/Jurkat/Dynamic/5. Jurkat_ON_OFF_10V_R1/Modified_masks"),

#     ("../organized_masks_data_7_7_2025/Jurkat/Dynamic/5. Jurkat_ON_OFF_10V_R3/10v_20s-40s_nd_091",
#      "../organized_masks_data_7_7_2025/Jurkat/Dynamic/5. Jurkat_ON_OFF_10V_R3/Modified_masks"),

#     ("../organized_masks_data_7_7_2025/SKBR3/Dynamic/6. SKBR3_ON_OFF_10V_R1/L6",
#      "../organized_masks_data_7_7_2025/SKBR3/Dynamic/6. SKBR3_ON_OFF_10V_R1/Modified_masks"),

#     ("../organized_masks_data_7_7_2025/SKBR3/Dynamic/6. SKBR3_ON_OFF_10V_R2/L4",
#      "../organized_masks_data_7_7_2025/SKBR3/Dynamic/6. SKBR3_ON_OFF_10V_R2/Modified_masks"),
    
#     ("../organized_masks_data_7_7_2025/SKBR3/Dynamic/6. SKBR3_ON_OFF_10V_R3/L3_001",
#      "../organized_masks_data_7_7_2025/SKBR3/Dynamic/6. SKBR3_ON_OFF_10V_R3/Modified_masks"),
    
#     ("../organized_masks_data_7_7_2025/SKBR3/Static/3. SKBR3_Static_R1/L3",
#      "../organized_masks_data_7_7_2025/SKBR3/Static/3. SKBR3_Static_R1/Modified_masks"),
    
#     ("../organized_masks_data_7_7_2025/SKBR3/Static/3. SKBR3_Static_R2/L5",
#      "../organized_masks_data_7_7_2025/SKBR3/Static/3. SKBR3_Static_R2/Modified_masks"),
    
#     ("../organized_masks_data_7_7_2025/SKBR3/Static/3. SKBR3_Static_R3/L2",
#      "../organized_masks_data_7_7_2025/SKBR3/Static/3. SKBR3_Static_R3/Modified_masks"),
# ]

# batch_size = 1
# batches = [folder_pairs[i:i+batch_size] for i in range(0, len(folder_pairs), batch_size)]

# model_path = None

# for stage_idx, folder_batch in enumerate(batches):
#     print(f"\n🧪 Training batch {stage_idx+1}/{len(batches)}")

#     all_images, all_masks, all_tns = [], [], []

#     for img_path, mask_path in folder_batch:
#         images, masks, tn_lists = load_image_mask_pairs(Path(img_path), Path(mask_path))
#         base_idx = len(all_images)
#         all_images.extend(images)
#         all_masks.extend(masks)
#         for i, tn_coords in enumerate(tn_lists):
#             for y, x in tn_coords:
#                 all_tns.append((base_idx + i, y, x))

#     print(f"✅ Batch {stage_idx+1}: {len(all_images)} images, {len(all_masks)} masks")

#     if model_path:
#         model = models.CellposeModel(gpu=True, pretrained_model=model_path)
#     else:
#         model = models.CellposeModel(gpu=True)

#     model_name = f"fine_model_new_{stage_idx+1}"
#     print(f"🚀 Training model: {model_name}")

#     model_path, train_losses, test_losses = train.train_seg(
#         model.net,
#         train_data=all_images,
#         train_labels=all_masks,
#         tn_coords=all_tns,
#         batch_size=1,
#         n_epochs=100,
#         learning_rate=1e-5,
#         weight_decay=0.1,
#         nimg_per_epoch=min(60, len(all_images)),
#         model_name=model_name
#     )

#     print(f"💾 Saved model from batch {stage_idx+1} to: {model_path}")

#     plt.figure(figsize=(10,5))
#     plt.plot(train_losses, label="Train Loss")
#     plt.plot(test_losses, label="Val Loss")
#     plt.xlabel("Epoch")
#     plt.ylabel("Loss")
#     plt.title(f"Loss Curve for {model_name}")
#     plt.legend()
#     plt.grid(True)
#     plt.savefig(f"loss_curve_{model_name}.png")
#     print(f"📈 Saved loss curve as: loss_curve_{model_name}.png")

# print(f"\n✅ All batches complete. Final model saved at: {model_path}")








#BEFORE I JOINED ALL THE FOLDERS INTO 1 AND SHUFFLED AND UNIFIED THEIR NAMES
# from cellpose import io, models, core, train
# from skimage.measure import label
# from pathlib import Path
# import numpy as np
# import re
# import matplotlib.pyplot as plt
# from skimage.transform import resize
# from sklearn.model_selection import train_test_split

# import os
# from skimage.color import label2rgb
# from skimage.io import imsave
# from skimage.util import img_as_ubyte
# from skimage.segmentation import find_boundaries
# import gc
# import torch


# # ------------------ Helper Functions ------------------ #
# def draw_square_on_image(img, y, x, size=7, color=(255, 0, 0)):
#     half = size // 2
#     y1 = max(0, y - half)
#     y2 = min(img.shape[0], y + half + 1)
#     x1 = max(0, x - half)
#     x2 = min(img.shape[1], x + half + 1)
#     img[y1:y2, x1:x2] = color

# def extract_index(filename):
#     match = re.search(r"(\d+)", filename.stem)
#     return int(match.group(1)) if match else None

# def bin_to_labels(mask):
#     threshold = np.max(mask) * 0.5
#     binary = mask > threshold
#     labeled = label(binary.astype(np.uint8), connectivity=1)
#     return labeled

# def sample_true_negatives_from_resized_mask(resized_mask, num_samples=30):
#     tn_coords = []
#     bg_coords = np.argwhere(resized_mask == 0)
#     if len(bg_coords) > 0:
#         sampled = bg_coords[np.random.choice(len(bg_coords), min(num_samples, len(bg_coords)), replace=False)]
#         for yx in sampled:
#             tn_coords.append((yx[0], yx[1]))
#     return tn_coords


# # works with only cells or background
# '''
# def load_image_mask_pairs(image_dir, mask_dir, target_shape=(256, 256), num_tns=30):
#     image_files = sorted(image_dir.glob("t*.tif"))
#     mask_files = sorted(mask_dir.glob("Masks*.tif"))
#     # image_files = sorted(image_dir.glob("img_*.tif"))
#     # mask_files = sorted(mask_dir.glob("mask_*.tif"))

#     image_map = {extract_index(f): f for f in image_files}
#     mask_map = {extract_index(f): f for f in mask_files}

#     common_indices = sorted(set(image_map.keys()) & set(mask_map.keys()))
#     print(f"📂 {image_dir.name} → {len(common_indices)} matched pairs")

#     images, masks, tns_all = [], [], []

#     for i in common_indices:
#         try:
#             img = io.imread(image_map[i])
#             mask = bin_to_labels(io.imread(mask_map[i]))
#         except Exception as e:
#             print(f"❌ Skipping index {i} due to read error: {e}")
#             continue

#         resized_mask = resize(mask.astype(np.uint8), target_shape, order=0, preserve_range=True, anti_aliasing=False).astype(np.uint8)
#         tn_coords = sample_true_negatives_from_resized_mask(resized_mask, num_tns)

#         images.append(img)
#         masks.append(mask)
#         tns_all.append(tn_coords)

        
#         # # Debugging: Save resized mask and TN coordinates
#         # print(tn_coords)

#         # # Save GT overlay with TNs (for sanity check)
#         # debug_dir = Path("debug_outputs") / "final_overlay"
#         # debug_dir.mkdir(parents=True, exist_ok=True)

#         # if img.dtype == np.uint16 or img.max() > 1:
#         #     norm_img = ((img - img.min()) / (img.max() - img.min())).astype(np.float32)
#         # else:
#         #     norm_img = img.astype(np.float32)

#         # norm_img_rgb = np.stack([norm_img]*3, axis=-1) if norm_img.ndim == 2 else norm_img

#         # overlay_green = label2rgb(mask, image=norm_img_rgb, bg_label=0, alpha=0.4, colors=['lime'])

#         # overlay_uint8 = img_as_ubyte(np.clip(overlay_green, 0, 1))

#         # scale_y = mask.shape[0] / resized_mask.shape[0]
#         # scale_x = mask.shape[1] / resized_mask.shape[1]
#         # for y, x in tn_coords:
#         #     y_orig = int(y * scale_y)
#         #     x_orig = int(x * scale_x)
#         #     # if 0 <= y_orig < overlay_uint8.shape[0] and 0 <= x_orig < overlay_uint8.shape[1]:
#         #     #     overlay_uint8[y_orig, x_orig] = [255, 0, 0]  # Red TN marker
#         #     draw_square_on_image(overlay_uint8, y_orig, x_orig, size=7, color=(255, 0, 0))

#         # imsave(debug_dir / f"overlay_img_mask_tn_{i}.png", overlay_uint8)
        

#     return images, masks, tns_all
# '''

# # to work with the dead/chaining cells that should be excluded
# def load_image_mask_pairs(image_dir, mask_dir, exclude_mask_dir=None, target_shape=(256, 256), num_tns=30):
#     image_files = sorted(image_dir.glob("img*.tif"))
#     mask_files = sorted(mask_dir.glob("mask*.tif"))
#     exclude_files = sorted(exclude_mask_dir.glob("exclude*.tif")) if exclude_mask_dir else []

#     image_map = {extract_index(f): f for f in image_files}
#     mask_map = {extract_index(f): f for f in mask_files}
#     exclude_map = {extract_index(f): f for f in exclude_files} if exclude_mask_dir else {}

#     # common_indices = sorted(set(image_map.keys()) & set(mask_map.keys()))[:70]
#     common_indices = sorted(set(image_map.keys()) & set(mask_map.keys()))

#     print(f"📂 {image_dir.name} → {len(common_indices)} matched pairs")

#     images, masks, tns_all = [], [], []

#     for i in common_indices:
#         try:
#             img = io.imread(image_map[i])
#             main_mask = bin_to_labels(io.imread(mask_map[i]))

#             if i in exclude_map:
#                 exclude_mask = bin_to_labels(io.imread(exclude_map[i]))
#                 # Exclude dead cells/chaining by zeroing out overlapping labels
#                 main_mask[exclude_mask > 0] = 0
#             else:
#                 exclude_mask = None
#         except Exception as e:
#             print(f"❌ Skipping index {i} due to read error: {e}")
#             continue

#         resized_mask = resize(main_mask.astype(np.uint8), target_shape, order=0, preserve_range=True, anti_aliasing=False).astype(np.uint8)
#         tn_coords = sample_true_negatives_from_resized_mask(resized_mask, num_tns)

#         images.append(img)
#         masks.append(main_mask)
#         tns_all.append(tn_coords)

#     return images, masks, tns_all


# # ------------------ Paths ------------------ #
# # folder_pairs = [
# #     ("../organized_masks_data_7_7_2025/Jurkat/Dynamic/5. Jurkat_ON_OFF_10V_R1/10v_20s-40s_nd_089",
# #      "../organized_masks_data_7_7_2025/Jurkat/Dynamic/5. Jurkat_ON_OFF_10V_R1/Modified_masks"),

# #     ("../organized_masks_data_7_7_2025/Jurkat/Dynamic/5. Jurkat_ON_OFF_10V_R3/10v_20s-40s_nd_091",
# #      "../organized_masks_data_7_7_2025/Jurkat/Dynamic/5. Jurkat_ON_OFF_10V_R3/Modified_masks"),

# #     ("../organized_masks_data_7_7_2025/SKBR3/Dynamic/6. SKBR3_ON_OFF_10V_R1/L6",
# #      "../organized_masks_data_7_7_2025/SKBR3/Dynamic/6. SKBR3_ON_OFF_10V_R1/Modified_masks"),

# #     ("../organized_masks_data_7_7_2025/SKBR3/Dynamic/6. SKBR3_ON_OFF_10V_R2/L4",
# #      "../organized_masks_data_7_7_2025/SKBR3/Dynamic/6. SKBR3_ON_OFF_10V_R2/Modified_masks"),
    
# #     ("../organized_masks_data_7_7_2025/SKBR3/Dynamic/6. SKBR3_ON_OFF_10V_R3/L3_001",
# #      "../organized_masks_data_7_7_2025/SKBR3/Dynamic/6. SKBR3_ON_OFF_10V_R3/Modified_masks"),
    
# #     ("../organized_masks_data_7_7_2025/SKBR3/Static/3. SKBR3_Static_R1/L3",
# #      "../organized_masks_data_7_7_2025/SKBR3/Static/3. SKBR3_Static_R1/Modified_masks"),
    
# #     ("../organized_masks_data_7_7_2025/SKBR3/Static/3. SKBR3_Static_R2/L5",
# #      "../organized_masks_data_7_7_2025/SKBR3/Static/3. SKBR3_Static_R2/Modified_masks"),
    
# #     ("../organized_masks_data_7_7_2025/SKBR3/Static/3. SKBR3_Static_R3/L2",
# #      "../organized_masks_data_7_7_2025/SKBR3/Static/3. SKBR3_Static_R3/Modified_masks"),
# # ]

# # folder_pairs = [
# #     ("../filtered_DEP_data/2. Jurkat_Static_R3/Images/nd_077_right 3",
# #      "../filtered_DEP_data/2. Jurkat_Static_R3/Modified Masks")]

# folder_pairs = [("../flattened_dataset/all_images",
#      "../flattened_dataset/all_masks")]

# # folder_pairs = [("../flattened_dataset/all_images",
# #      "../flattened_dataset/all_masks",
# #     "../flattened_dataset/exclude_masks")]


# batch_size = 1
# batches = [folder_pairs[i:i+batch_size] for i in range(0, len(folder_pairs), batch_size)]


# model_path = None

# for stage_idx, folder_batch in enumerate(batches):
#     print(f"\n🧪 Training batch {stage_idx+1}/{len(batches)}")

#     all_images, all_masks, all_tns = [], [], []


#     # for img_path, mask_path, exclude_path in folder_batch:
#     #     images, masks, tn_lists = load_image_mask_pairs(Path(img_path), Path(mask_path), Path(exclude_path))

#     for img_path, mask_path in folder_batch:
#         images, masks, tn_lists = load_image_mask_pairs(Path(img_path), Path(mask_path))


#         base_idx = len(all_images)
#         all_images.extend(images)
#         all_masks.extend(masks)
#         for i, tn_coords in enumerate(tn_lists):
#             for y, x in tn_coords:
#                 all_tns.append((base_idx + i, y, x))

#     print(f"✅ Batch {stage_idx+1}: {len(all_images)} images, {len(all_masks)} masks")

#     if model_path:
#         model = models.CellposeModel(gpu=True, pretrained_model=model_path)
#     else:
#         model = models.CellposeModel(gpu=True)

#     model_name = f"fine_model_flattened_{stage_idx+1}"
#     print(f"🚀 Training model: {model_name}")

#     # model_path, train_losses, test_losses = train.train_seg(
#     #     model.net,
#     #     train_data=all_images,
#     #     train_labels=all_masks,
#     #     tn_coords=all_tns,
#     #     batch_size=1,
#     #     n_epochs=100,
#     #     learning_rate=1e-5,
#     #     weight_decay=0.1,
#     #     nimg_per_epoch=min(60, len(all_images)),
#     #     model_name=model_name
#     # )

#     # ADDING VALIDATION SPLIT
#     # Split images, masks, and TNs into training and validation sets
#     X_train, X_val, y_train, y_val = train_test_split(all_images, all_masks, test_size=0.2, random_state=42)

#     # Adjust TNs for training and validation
#     tn_train = [(i, y, x) for (i, y, x) in all_tns if i < len(X_train)]
#     tn_val   = [(i - len(X_train), y, x) for (i, y, x) in all_tns if i >= len(X_train)]

#     # Run training with validation data
#     model_path, train_losses, test_losses = train.train_seg(
#         model.net,
#         train_data=X_train,
#         train_labels=y_train,
#         test_data=X_val,
#         test_labels=y_val,
#         tn_coords=tn_train,
#         test_tn_coords=tn_val,
#         batch_size=1,
#         n_epochs=70,
#         learning_rate=1e-5,
#         weight_decay=0.1,
#         nimg_per_epoch=min(60, len(X_train)),
#         model_name=model_name
#     )




#     print(f"💾 Saved model from batch {stage_idx+1} to: {model_path}")

#     # === Run inference on training images ===
#     # print("🔍 Running inference on training images...")
#     # preds, _, _ = model.eval(all_images, channels=[[0, 0]], batch_size=1)

#     # debug_overlay_dir = Path("debug_outputs") / f"overlay_predictions_batch_{stage_idx+1}"
#     # debug_overlay_dir.mkdir(parents=True, exist_ok=True)

#     # for i, img in enumerate(all_images):
#     #     gt_mask = all_masks[i]
#     #     pred_mask = preds[i]

#     #     if img.dtype == np.uint16 or img.max() > 1:
#     #         norm_img = ((img - img.min()) / (img.max() - img.min())).astype(np.float32)
#     #     else:
#     #         norm_img = img.astype(np.float32)

#     #     norm_img_rgb = np.stack([norm_img]*3, axis=-1) if norm_img.ndim == 2 else norm_img

#     #     overlay_pred = label2rgb(pred_mask, image=norm_img_rgb, bg_label=0, alpha=0.4, colors=['red'])
#     #     overlay_gt = label2rgb(gt_mask, image=overlay_pred, bg_label=0, alpha=0.4, colors=['green'])

#     #     overlay_uint8 = img_as_ubyte(np.clip(overlay_gt, 0, 1))
#     #     imsave(debug_overlay_dir / f"debug_overlay_{i}.png", overlay_uint8)
#     #     print(f"🖼️ Saved: debug_overlay_{i}.png")

#     # === Plot loss ===
#     plt.figure(figsize=(10,5))
#     plt.plot(train_losses, label="Train Loss")
#     plt.plot(test_losses, label="Val Loss")
#     plt.xlabel("Epoch")
#     plt.ylabel("Loss")
#     plt.title(f"Loss Curve for {model_name}")
#     plt.legend()
#     plt.grid(True)

#     # 🔒 Ensure folder exists
#     loss_dir = Path("loss_curves")
#     loss_dir.mkdir(parents=True, exist_ok=True)
#     # 💾 Save inside loss_curves/
#     plt.savefig(loss_dir / f"loss_curve_{model_name}.png")
#     plt.close('all')  # free memory
    
#     print(f"📈 Saved loss curve as: loss_curve_{model_name}.png")

# print(f"\n✅ All batches complete. Final model saved at: {model_path}")

# print("🔍 Peak GPU memory used (MB):", torch.cuda.max_memory_allocated() / 1e6)












#WITH BATCHES: REMOVED BECAUSE RETRAINING DOESNT SEEM TO WORK, THE 2ND MODEL ONWARDS IS HORRIBLE
# batch_size = 5
# total_batches = (len(combined_images) + batch_size - 1) // batch_size


# model_path = None  # no pretrained model for first batch

# for stage_idx in range(total_batches):
#     start_idx = stage_idx * batch_size
#     end_idx = min((stage_idx + 1) * batch_size, len(combined_images))

#     images = combined_images[start_idx:end_idx]
#     masks = combined_masks[start_idx:end_idx]
#     tns = [(i - start_idx, y, x) for (i, y, x) in combined_tns if start_idx <= i < end_idx]

#     print(f"\n🧪 Training batch {stage_idx+1}/{total_batches}: {len(images)} images")

#     # Split into train and val
#     X_train, X_val, y_train, y_val = train_test_split(images, masks, test_size=0.2, random_state=42)
#     tn_train = [(i, y, x) for (i, y, x) in tns if i < len(X_train)]
#     tn_val = [(i - len(X_train), y, x) for (i, y, x) in tns if i >= len(X_train)]

#     # Load previous model or initialize new
#     model = models.CellposeModel(gpu=True, pretrained_model=model_path) if model_path else models.CellposeModel(gpu=True)

#     model_name = f"fine_model_chain_chunk_{stage_idx+1}"
#     print(f"🚀 Training model: {model_name}")

#     model_path, train_losses, test_losses = train.train_seg(
#         model.net,
#         train_data=X_train,
#         train_labels=y_train,
#         test_data=X_val,
#         test_labels=y_val,
#         tn_coords=tn_train,
#         test_tn_coords=tn_val,
#         batch_size=1,
#         n_epochs=70,
#         learning_rate=1e-5,
#         weight_decay=0.1,
#         nimg_per_epoch=min(30, len(X_train)),
#         model_name=model_name
#     )

#     # Plot loss curves and save
#     plt.figure(figsize=(10,5))
#     plt.plot(train_losses, label="Train Loss")
#     plt.plot(test_losses, label="Val Loss")
#     plt.xlabel("Epoch")
#     plt.ylabel("Loss")
#     plt.title(f"Loss Curve for {model_name}")
#     plt.legend()
#     plt.grid(True)

#     loss_dir = Path("loss_curves")
#     loss_dir.mkdir(parents=True, exist_ok=True)
#     plt.savefig(loss_dir / f"loss_curve_{model_name}.png")
#     plt.close('all')
#     del model, images, masks, tns, X_train, X_val, y_train, y_val
#     torch.cuda.empty_cache()
#     gc.collect()

#     process = psutil.Process(os.getpid())
#     print(f"🧠 RAM used after cleanup: {process.memory_info().rss / 256**3:.2f} GB")
#     print(f"🖥️ GPU used after cleanup: {torch.cuda.memory_allocated() / 256**3:.2f} GB")


#     print(f"📈 Saved loss curve for batch {stage_idx+1} as: loss_curve_{model_name}.png")
#     print(f"💾 Model saved to: {model_path}")

    



#TRAIN ON ALL AT ONCE
# # Initialize a new model (no pretraining)
# model = models.CellposeModel(gpu=True)

# model_name = "fine_model_full" #NO CHAINING
# # model_name = "fine_model_chain_full" #CHAINING
# print(f"\n🚀 Training model: {model_name} on full dataset with {len(combined_images)} images")

# # Train once on the entire dataset
# model_path, train_losses, test_losses = train.train_seg(
#     model.net,
#     train_data=X_train,
#     train_labels=y_train,
#     test_data=X_val,
#     test_labels=y_val,
#     tn_coords=tn_train,
#     test_tn_coords=tn_val,
#     batch_size=1,
#     n_epochs=70,
#     learning_rate=1e-5,
#     weight_decay=0.1,
#     nimg_per_epoch=min(30, len(X_train)),
#     model_name=model_name
# )

# # Plot and save loss curve
# plt.figure(figsize=(10, 5))
# plt.plot(train_losses, label="Train Loss")
# plt.plot(test_losses, label="Val Loss")
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.title(f"Loss Curve for {model_name}")
# plt.legend()
# plt.grid(True)

# loss_dir = Path("loss_curves")
# loss_dir.mkdir(parents=True, exist_ok=True)
# plt.savefig(loss_dir / f"loss_curve_{model_name}.png")
# plt.close('all')

# torch.cuda.empty_cache()
# gc.collect()

# process = psutil.Process(os.getpid())
# print(f"🧠 RAM used after cleanup: {process.memory_info().rss / 256**3:.2f} GB")
# print(f"🖥️ GPU used after cleanup: {torch.cuda.memory_allocated() / 256**3:.2f} GB")

# print(f"📈 Saved loss curve as: loss_curve_{model_name}.png")
# print(f"💾 Model saved to: {model_path}")





#Train in chunks without reinitializing model each time
# chunk_size = 70  # Adjust based on your memory capacity
# num_chunks = (len(X_train) + chunk_size - 1) // chunk_size

# model = models.CellposeModel(gpu=True)
# model_name = "fine_model_full_incremental"

# all_train_losses, all_val_losses = [], []

# for chunk_id in range(num_chunks):
#     start = chunk_id * chunk_size
#     end = min(start + chunk_size, len(X_train))
    
#     X_chunk = X_train[start:end]
#     y_chunk = y_train[start:end]

#     tn_chunk = [(i - start, y, x) for (i, y, x) in tn_train if start <= i < end]

#     print(f"\n🧪 Training chunk {chunk_id+1}/{num_chunks} → {len(X_chunk)} samples")

#     model_path, train_losses, val_losses = train.train_seg(
#         model.net,
#         train_data=X_chunk,
#         train_labels=y_chunk,
#         test_data=X_val,
#         test_labels=y_val,
#         tn_coords=tn_chunk,
#         test_tn_coords=tn_val,
#         batch_size=1,
#         n_epochs=70,
#         learning_rate=1e-5,
#         weight_decay=0.1,
#         nimg_per_epoch=min(30, len(X_chunk)),
#         model_name=model_name
#     )

#     all_train_losses.extend(train_losses)
#     all_val_losses.extend(val_losses)

#     torch.cuda.empty_cache()
#     gc.collect()

# # Plot final loss curve
# plt.figure(figsize=(10, 5))
# plt.plot(all_train_losses, label="Train Loss")
# plt.plot(all_val_losses, label="Val Loss")
# plt.xlabel("Epochs (accumulated)")
# plt.ylabel("Loss")
# plt.title(f"Loss Curve for {model_name}")
# plt.legend()
# plt.grid(True)

# loss_dir = Path("loss_curves")
# loss_dir.mkdir(parents=True, exist_ok=True)
# plt.savefig(loss_dir / f"loss_curve_{model_name}.png")
# plt.close('all')

# torch.cuda.empty_cache()
# gc.collect()

# process = psutil.Process(os.getpid())
# print(f"🧠 RAM used after cleanup: {process.memory_info().rss / 256**3:.2f} GB")
# print(f"🖥️ GPU used after cleanup: {torch.cuda.memory_allocated() / 256**3:.2f} GB")
# print(f"📈 Saved loss curve as: loss_curve_{model_name}.png")
# print(f"💾 Final model saved to: {model_path}")





















#SHUFFLED UNIFIED FOLDER
# from cellpose import io, models, core, train
# from skimage.measure import label
# from pathlib import Path
# import numpy as np
# import re
# import matplotlib.pyplot as plt
# from skimage.transform import resize
# from sklearn.model_selection import train_test_split

# import os
# from skimage.color import label2rgb
# from skimage.io import imsave
# from skimage.util import img_as_ubyte
# from skimage.segmentation import find_boundaries
# import gc
# import torch
# import psutil


# # ------------------ Helper Functions ------------------ #
# def draw_square_on_image(img, y, x, size=7, color=(255, 0, 0)):
#     half = size // 2
#     y1 = max(0, y - half)
#     y2 = min(img.shape[0], y + half + 1)
#     x1 = max(0, x - half)
#     x2 = min(img.shape[1], x + half + 1)
#     img[y1:y2, x1:x2] = color

# def extract_index(filename):
#     match = re.search(r"(\d+)", filename.stem)
#     return int(match.group(1)) if match else None

# def bin_to_labels(mask):
#     threshold = np.max(mask) * 0.5
#     binary = mask > threshold
#     labeled = label(binary.astype(np.uint8), connectivity=1)
#     return labeled

# def sample_true_negatives_from_resized_mask(resized_mask, num_samples=30):
#     tn_coords = []
#     bg_coords = np.argwhere(resized_mask == 0)
#     if len(bg_coords) > 0:
#         sampled = bg_coords[np.random.choice(len(bg_coords), min(num_samples, len(bg_coords)), replace=False)]
#         for yx in sampled:
#             tn_coords.append((yx[0], yx[1]))
#     return tn_coords



# # to work with the dead/chaining cells that should be excluded
# def load_image_mask_pairs(image_dir, mask_dir, exclude_mask_dir=None, target_shape=(256, 256), num_tns=30):
#     #NO CHAINING
#     # image_files = sorted(image_dir.glob("img*.tif"))
#     # mask_files = sorted(mask_dir.glob("mask*.tif"))    

#     #CHAINING
#     image_files = sorted(image_dir.glob("t*.tif"))
#     mask_files = sorted(mask_dir.glob("masks*.tif"))


#     exclude_files = sorted(exclude_mask_dir.glob("exclude_*.tif")) if exclude_mask_dir else []


#     image_map = {extract_index(f): f for f in image_files}
#     mask_map = {extract_index(f): f for f in mask_files}
#     exclude_map = {extract_index(f): f for f in exclude_files} if exclude_mask_dir else {}

#     common_indices = sorted(set(image_map.keys()) & set(mask_map.keys()))[:280]
#     # common_indices = sorted(set(image_map.keys()) & set(mask_map.keys()))

#     print(f"📂 {image_dir.name} → {len(common_indices)} matched pairs")

#     images, masks, tns_all = [], [], []

#     for i in common_indices:
#         try:
#             img = io.imread(image_map[i])
#             main_mask = bin_to_labels(io.imread(mask_map[i]))
            
#             # num_unique = len(np.unique(main_mask)) - 1  # exclude background (label 0)
#             # print(f"    → Detected {num_unique} mask regions")
#             # print(f"🔍 Processing index {i} → Image shape: {img.shape}, Mask shape: {main_mask.shape}")

#             if i in exclude_map:
#                 exclude_mask = bin_to_labels(io.imread(exclude_map[i]))
#                 # Exclude dead cells/chaining by zeroing out overlapping labels
#                 main_mask[exclude_mask > 0] = 0
#             else:
#                 exclude_mask = None
#         except Exception as e:
#             print(f"❌ Skipping index {i} due to read error: {e}")
#             continue

#         resized_mask = resize(main_mask.astype(np.uint8), target_shape, order=0, preserve_range=True, anti_aliasing=False).astype(np.uint8)
#         tn_coords = sample_true_negatives_from_resized_mask(resized_mask, num_tns)



#         # images.append(img)
#         # masks.append(main_mask)
#         # tns_all.append(tn_coords)
#         patch_size = 1000  # or 256 if you want smaller patches
#         stride = 1000       # or use patch_size // 2 for overlapping

#         '''
#         🔍 Processing index 229 → Image shape: (3264, 4908), Mask shape: (3264, 4908)
#         <tifffile.TiffPages @32039432> invalid offset to first page 32039432
#             → Detected 10 mask regions
#         🔍 Processing index 230 → Image shape: (0,), Mask shape: (3264, 4908)
#         ❌ Skipping index 230 due to invalid image shape: (0,)
#             → Detected 14 mask regions
#         '''
#         if img.ndim != 2 or img.shape[0] == 0 or img.shape[1] == 0:
#             print(f"❌ Skipping index {i} due to invalid image shape: {img.shape}")
#             continue
#         H, W = img.shape

#         for y in range(0, H - patch_size + 1, stride):
#             for x in range(0, W - patch_size + 1, stride):
#                 patch_img = img[y:y + patch_size, x:x + patch_size]
#                 patch_mask = main_mask[y:y + patch_size, x:x + patch_size]

#                 # Skip completely empty patches
#                 if np.sum(patch_mask) == 0:
#                     continue

#                 resized_mask = resize(patch_mask.astype(np.uint8), target_shape, order=0, preserve_range=True, anti_aliasing=False).astype(np.uint8)
#                 tn_coords_patch = sample_true_negatives_from_resized_mask(resized_mask, num_tns)

#                 images.append(patch_img)
#                 masks.append(patch_mask)
#                 tns_all.append(tn_coords_patch)



#     return images, masks, tns_all

# #NO CHAINING
# # folder_pairs = [("../flattened_dataset/all_images",
# #      "../flattened_dataset/all_masks")]

# #CHAINING
# folder_pairs = [
#     ("../filtered_DEP_data/3. SKBR3_Static_R1/Images/L3", 
#      "../filtered_DEP_data/3. SKBR3_Static_R1/Modified_Masks", 
#      "../filtered_DEP_data/3. SKBR3_Static_R1/Chaining"),

#     ("../filtered_DEP_data/3. SKBR3_Static_R2/Images/L5", 
#      "../filtered_DEP_data/3. SKBR3_Static_R2/Modified Masks", 
#      "../filtered_DEP_data/3. SKBR3_Static_R2/Chaining"),

#     ("../filtered_DEP_data/3. SKBR3_Static_R3/Images/L2", 
#      "../filtered_DEP_data/3. SKBR3_Static_R3/Modified Masks", 
#      "../filtered_DEP_data/3. SKBR3_Static_R3/Chaining"),

#     ("../filtered_DEP_data/5. Jurkat_ON_OFF_10V_R1/Images/10v_20s-40s_nd_089", 
#      "../filtered_DEP_data/5. Jurkat_ON_OFF_10V_R1/Modified_masks", 
#      "../filtered_DEP_data/5. Jurkat_ON_OFF_10V_R1/Chaining"),

#     ("../filtered_DEP_data/5. Jurkat_ON_OFF_10V_R2/Images/10v_20s-40s_nd_090", 
#      "../filtered_DEP_data/5. Jurkat_ON_OFF_10V_R2/Modified_masks", 
#      "../filtered_DEP_data/5. Jurkat_ON_OFF_10V_R2/Chaining"),
# ]

# # Combine all image-mask pairs from all folder pairs
# combined_images, combined_masks, combined_tns = [], [], []

# print("🔍 [0] Initializing model...")
# model = models.CellposeModel(gpu=True)
# # model_name = "fine_model_all_without_chaining"
# model_name = "fine_model_all_chaining"

# print("🔍 [1] Starting data loading...")

# #NO CHAINING
# # for img_path, mask_path in folder_pairs:
# #     print(f"🔍 [1.1] Loading from: {img_path}")
# #     images, masks, tn_lists = load_image_mask_pairs(Path(img_path), Path(mask_path))

# #CHAINING
# for img_path, mask_path, exclude_path in folder_pairs:
#     print(f"🔍 [1.1] Loading from: {img_path}")
#     images, masks, tn_lists = load_image_mask_pairs(Path(img_path), Path(mask_path), Path(exclude_path))

#     base_idx = len(combined_images)
#     combined_images.extend(images)
#     combined_masks.extend(masks)
#     for i, tn_coords in enumerate(tn_lists):
#         for y, x in tn_coords:
#             combined_tns.append((base_idx + i, y, x))

# print(f"\n📊 Total loaded: {len(combined_images)} images")
# print("🔍 [2] Splitting into train/val...")

# # Split all data into train and val
# X_train, X_val, y_train, y_val = train_test_split(combined_images, combined_masks, test_size=0.2, random_state=42)

# tn_train = [(i, y, x) for (i, y, x) in combined_tns if i < len(X_train)]
# tn_val = [(i - len(X_train), y, x) for (i, y, x) in combined_tns if i >= len(X_train)]

# print(f"🔍 [3] Training set: {len(X_train)} | Val set: {len(X_val)}")



# print("🔍 [3.5] Cleaning up before model init...")
# torch.cuda.empty_cache()
# gc.collect()

# process = psutil.Process(os.getpid())
# print(f"🧠 RAM: {process.memory_info().rss / 256**3:.2f} GB | 🖥️ GPU: {torch.cuda.memory_allocated() / 256**3:.2f} GB")




# print(f"\n🚀 Training model on full dataset with {len(X_train)} images")
# print(f"[🔍] RAM before train: {psutil.virtual_memory().used / 1e9:.2f} GB")

# # print("\n🧪 Checking number of mask regions per training image:")
# # for i, mask in enumerate(y_train):
# #     unique_labels = np.unique(mask)
# #     n_masks = len(unique_labels) - (1 if 0 in unique_labels else 0)
# #     print(f"  Train image {i}: {n_masks} masks")

# # print("\n🧪 Checking number of mask regions per validation image:")
# # for i, mask in enumerate(y_val):
# #     unique_labels = np.unique(mask)
# #     n_masks = len(unique_labels) - (1 if 0 in unique_labels else 0)
# #     print(f"  Val image {i}: {n_masks} masks")


# model_path, train_losses, test_losses = train.train_seg(
#     model.net,
#     train_data=X_train,
#     train_labels=y_train,
#     test_data=X_val,
#     test_labels=y_val,
#     tn_coords=tn_train,
#     test_tn_coords=tn_val,
#     batch_size=1,
#     n_epochs=70,                      # increase epochs if nimg_per_epoch is small
#     learning_rate=1e-5,
#     weight_decay=0.1,
#     nimg_per_epoch=30,                # limits memory usage by simulating small batches
#     model_name=model_name,
# )
# print("✅ [6] Training complete — saving loss curves...")


# # Plot and save loss curve
# plt.figure(figsize=(10, 5))
# plt.plot(train_losses, label="Train Loss")
# plt.plot(test_losses, label="Val Loss")
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.title(f"Loss Curve for {model_name}")
# plt.legend()
# plt.grid(True)

# loss_dir = Path("loss_curves")
# loss_dir.mkdir(parents=True, exist_ok=True)
# plt.savefig(loss_dir / f"loss_curve_{model_name}.png")
# plt.close('all')

# torch.cuda.empty_cache()
# gc.collect()

# process = psutil.Process(os.getpid())
# print(f"🧠 RAM used after cleanup: {process.memory_info().rss / 256**3:.2f} GB")
# print(f"🖥️ GPU used after cleanup: {torch.cuda.memory_allocated() / 256**3:.2f} GB")

# print(f"📈 Saved loss curve as: loss_curve_{model_name}.png")
# print(f"💾 Model saved to: {model_path}")










# ========== CONFIG ==========

#NO CHAINING
# model_name = "fine_model_all_without_chaining"
# img_pattern = "img"     
# mask_pattern = "mask"   
# folder_pairs = [("../flattened_dataset/all_images",
#      "../flattened_dataset/all_masks")]


#CHAINING
# model_name = "fine_model_all_chaining"
model_name = "fine_model_all_chaining_tn"
img_pattern = "t"   
mask_pattern = "masks"   
folder_pairs = [
    ("../filtered_DEP_data/3. SKBR3_Static_R1/Images/L3", 
     "../filtered_DEP_data/3. SKBR3_Static_R1/Modified_Masks", 
     "../filtered_DEP_data/3. SKBR3_Static_R1/Chaining"),

    ("../filtered_DEP_data/3. SKBR3_Static_R2/Images/L5", 
     "../filtered_DEP_data/3. SKBR3_Static_R2/Modified Masks", 
     "../filtered_DEP_data/3. SKBR3_Static_R2/Chaining"),

    ("../filtered_DEP_data/3. SKBR3_Static_R3/Images/L2", 
     "../filtered_DEP_data/3. SKBR3_Static_R3/Modified Masks", 
     "../filtered_DEP_data/3. SKBR3_Static_R3/Chaining"),

    ("../filtered_DEP_data/5. Jurkat_ON_OFF_10V_R1/Images/10v_20s-40s_nd_089", 
     "../filtered_DEP_data/5. Jurkat_ON_OFF_10V_R1/Modified_masks", 
     "../filtered_DEP_data/5. Jurkat_ON_OFF_10V_R1/Chaining"),

    ("../filtered_DEP_data/5. Jurkat_ON_OFF_10V_R2/Images/10v_20s-40s_nd_090", 
     "../filtered_DEP_data/5. Jurkat_ON_OFF_10V_R2/Modified_masks", 
     "../filtered_DEP_data/5. Jurkat_ON_OFF_10V_R2/Chaining"),
]




import os, gc, re
import torch
import psutil
import numpy as np
from pathlib import Path
from skimage.measure import label, regionprops
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from cellpose import io, models, train



# ========== HELPERS ==========
def generate_patches(mask, patch_size=256, stride=128, num_tns=5):
    """
    Return coords: list of ('fg'|'bg', y_center, x_center).
    Centers are for intended crop centers; we clamp to bounds later.
    """
    coords = []
    h, w = mask.shape
    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            patch = mask[y:y+patch_size, x:x+patch_size]
            has_cells = np.max(patch) > 0
            if has_cells:
                coords.append(('fg', y + patch_size // 2, x + patch_size // 2))
            elif len([c for c in coords if c[0] == 'bg']) < num_tns:
                coords.append(('bg', y + patch_size // 2, x + patch_size // 2))
    return coords

def extract_index(filename):
    m = re.search(r"(\d+)", filename.stem)
    return int(m.group(1)) if m else None

def bin_to_labels(mask):
    binary = mask > (np.max(mask) * 0.5)
    return label(binary.astype(np.uint8), connectivity=1)

def is_valid_mask(lbl, min_fg_pixels=32, min_region_area=16):
    """
    lbl: 2D integer label map (H,W), 0=bg.
    Reject masks that are empty or have only tiny regions.
    """
    if lbl.ndim != 2:
        return False
    if lbl.max() == 0:
        return False
    fg = np.count_nonzero(lbl)
    if fg < min_fg_pixels:
        return False
    for r in regionprops(lbl):
        if r.area >= min_region_area:
            return True
    return False

# ========== DATA LOADING ==========
def load_image_mask_pairs(image_dir, mask_dir, exclude_mask_dir):
    image_dir, mask_dir = Path(image_dir), Path(mask_dir)
    exclude_mask_dir = Path(exclude_mask_dir) if exclude_mask_dir else None

    image_files = sorted(image_dir.glob(f"{img_pattern}*.tif"))
    mask_files = sorted(mask_dir.glob(f"{mask_pattern}*.tif"))
    exclude_files = sorted(exclude_mask_dir.glob("exclude_*.tif")) if exclude_mask_dir else []

    image_map = {extract_index(f): f for f in image_files}
    mask_map = {extract_index(f): f for f in mask_files}
    exclude_map = {extract_index(f): f for f in exclude_files}

    common_ids = sorted(set(image_map) & set(mask_map))

    images, masks, tns_all = [], [], []
    print(f"📂 {image_dir.name} → {len(common_ids)} matched pairs")

    for idx in common_ids:
        try:
            img = io.imread(image_map[idx])
            mask = bin_to_labels(io.imread(mask_map[idx]))
            exclude_mask = bin_to_labels(io.imread(exclude_map[idx])) if idx in exclude_map else None

            if img.ndim != 2 or mask.shape != img.shape:
                continue

            # Apply chaining exclusion
            if exclude_mask is not None:
                mask[exclude_mask > 0] = 0

            coords = generate_patches(mask, patch_size=256, stride=128, num_tns=30)

            for label_tag, y, x in coords:
                # Centered 256×256 crop clamped to bounds
                h, w = img.shape
                patch_size = 256
                half = patch_size // 2
                y0 = min(max(0, y - half), max(0, h - patch_size))
                x0 = min(max(0, x - half), max(0, w - patch_size))

                patch_img = img[y0:y0+patch_size, x0:x0+patch_size]
                patch_mask = mask[y0:y0+patch_size, x0:x0+patch_size]

                # Require a healthy labeled mask
                if not is_valid_mask(patch_mask):
                    continue

                # Make contiguous, correct dtype, and add channel: (1,H,W)
                patch_mask = np.ascontiguousarray(patch_mask.astype(np.int32))[None, :, :]
                patch_img  = np.ascontiguousarray(patch_img)

                images.append(patch_img)
                masks.append(patch_mask)
                patch_idx = len(images) - 1

                # TNs from chaining mask, if any
                if exclude_mask is not None:
                    patch_excl = exclude_mask[y0:y0+patch_size, x0:x0+patch_size]
                    chaining_coords = np.argwhere(patch_excl > 0)
                    if chaining_coords.size > 0:
                        tns_all.append([(patch_idx, int(cy), int(cx)) for cy, cx in chaining_coords])
                    else:
                        tns_all.append([])
                else:
                    tns_all.append([])

        except Exception as e:
            print(f"❌ Skipping index {idx} due to error: {e}")
            continue

    return images, masks, tns_all

# ========== MAIN ==========
model = models.CellposeModel(gpu=True)
all_images, all_masks, all_tns = [], [], []

for triplet in folder_pairs:
    if len(triplet) == 3:
        img_dir, mask_dir, excl_dir = triplet
    else:
        img_dir, mask_dir = triplet
        excl_dir = None

    imgs, msks, tns = load_image_mask_pairs(img_dir, mask_dir, excl_dir)

    base_idx = len(all_images)
    all_images.extend(imgs)
    all_masks.extend(msks)

    # flatten and offset TNs
    for coords in tns:
        for (img_idx, y, x) in coords:
            all_tns.append((base_idx + img_idx, y, x))

print(f"\n📊 Total images (pre-filter): {len(all_images)}")

# Remove any leftover empties (defensive) and remap TN indices
keep = [i for i, m in enumerate(all_masks) if np.max(m) > 0]
drop = set(range(len(all_masks))) - set(keep)
if drop:
    print(f"⚠️ Dropping {len(drop)} bg-only patches (no instances).")

idx_map = {old: new for new, old in enumerate(keep)}

all_images = [np.ascontiguousarray(all_images[i]) for i in keep]
all_masks  = [np.ascontiguousarray(all_masks[i].astype(np.int32)) for i in keep]  # keep (1,H,W)
all_tns    = [(idx_map[i], y, x) for (i, y, x) in all_tns if i in idx_map]

n_empty = sum(1 for m in all_masks if np.max(m) == 0)
print(f"✅ kept={len(all_masks)} | empty={n_empty} (must be 0)")
assert n_empty == 0, "Empty masks made it into training labels."

# Build per-image TN lists
tns_per_img = [[] for _ in range(len(all_images))]
for i, y, x in all_tns:
    if 0 <= i < len(tns_per_img):
        tns_per_img[i].append((int(y), int(x)))

# Split by indices to preserve TN alignment
idx = np.arange(len(all_images))
train_idx, val_idx = train_test_split(idx, test_size=0.2, random_state=42, shuffle=True)

X_train = [all_images[i] for i in train_idx]
y_train = [all_masks[i]  for i in train_idx]
X_val   = [all_images[i] for i in val_idx]
y_val   = [all_masks[i]  for i in val_idx]


# tn_train = [tns_per_img[i] for i in train_idx]   # list-of-lists of (y,x)
# tn_val   = [tns_per_img[i] for i in val_idx]

# Flat list of (i, y, x), where i is the index within X_train / X_val
tn_train = []
for local_i, global_i in enumerate(train_idx):
    for (y, x) in tns_per_img[global_i]:
        tn_train.append((local_i, int(y), int(x)))
tn_val = []
for local_i, global_i in enumerate(val_idx):
    for (y, x) in tns_per_img[global_i]:
        tn_val.append((local_i, int(y), int(x)))


# Cleanup
torch.cuda.empty_cache(); gc.collect()

# Memory log
p = psutil.Process(os.getpid())
print(f"🧠 RAM: {p.memory_info().rss / 1024**3:.2f} GB | GPU: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

# Sanity
print("Dataset sanity:")
print("  train:", len(X_train), "val:", len(X_val))
print("  example shapes:", X_train[0].shape, y_train[0].shape)  # (256,256), (1,256,256)
print("  min/mean FG pixels (train):",
      min(np.count_nonzero(m[0]) for m in y_train),
      int(np.mean([np.count_nonzero(m[0]) for m in y_train])))

# Train
print(f"\n🚀 Training model: {model_name}")
Path("loss_curves").mkdir(exist_ok=True)
model_path, train_losses, test_losses = train.train_seg(
    model.net,
    train_data=X_train,
    train_labels=y_train,
    test_data=X_val,
    test_labels=y_val,
    tn_coords=tn_train,        # aligned with X_train
    test_tn_coords=tn_val,     # aligned with X_val
    batch_size=1,
    n_epochs=70,
    learning_rate=1e-5,
    weight_decay=0.1,
    nimg_per_epoch=30,
    model_name=model_name
)

# Plot
plt.plot(train_losses, label="Train Loss")
plt.plot(test_losses, label="Val Loss")
plt.title(f"Loss Curve: {model_name}")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.legend()
plt.savefig(f"loss_curves/loss_curve_{model_name}.png")
plt.close()

print(f"\n📈 Saved loss curve to: loss_curve_{model_name}.png")
print(f"💾 Model saved at: {model_path}")

# ========== HELPERS ==========
# def generate_patches(mask, patch_size=256, stride=128, num_tns=5):
#     """
#     Generate coordinates of multiple patches from an image.
#     Returns a list of tuples: ('fg'|'bg', y_center, x_center).
#     """
#     coords = []
#     h, w = mask.shape

#     for y in range(0, h - patch_size + 1, stride):
#         for x in range(0, w - patch_size + 1, stride):
#             patch = mask[y:y+patch_size, x:x+patch_size]
#             has_cells = np.max(patch) > 0
#             if has_cells:
#                 coords.append(('fg', y + patch_size // 2, x + patch_size // 2))
#             elif len([c for c in coords if c[0] == 'bg']) < num_tns:
#                 coords.append(('bg', y + patch_size // 2, x + patch_size // 2))
#     return coords

# def extract_index(filename):
#     match = re.search(r"(\d+)", filename.stem)
#     return int(match.group(1)) if match else None

# def bin_to_labels(mask):
#     binary = mask > (np.max(mask) * 0.5)
#     return label(binary.astype(np.uint8), connectivity=1)

# # ========== DATA LOADING ==========
# def load_image_mask_pairs(image_dir, mask_dir, exclude_mask_dir):
#     image_dir, mask_dir = Path(image_dir), Path(mask_dir)
#     exclude_mask_dir = Path(exclude_mask_dir) if exclude_mask_dir else None

#     image_files = sorted(image_dir.glob(f"{img_pattern}*.tif"))
#     mask_files = sorted(mask_dir.glob(f"{mask_pattern}*.tif"))
#     exclude_files = sorted(exclude_mask_dir.glob("exclude_*.tif")) if exclude_mask_dir else []

#     image_map = {extract_index(f): f for f in image_files}
#     mask_map = {extract_index(f): f for f in mask_files}
#     exclude_map = {extract_index(f): f for f in exclude_files}

#     common_ids = sorted(set(image_map) & set(mask_map))

#     images, masks, tns_all = [], [], []
#     print(f"📂 {image_dir.name} → {len(common_ids)} matched pairs")

#     for idx in common_ids:
#         try:
#             img = io.imread(image_map[idx])
#             mask = bin_to_labels(io.imread(mask_map[idx]))
#             exclude_mask = bin_to_labels(io.imread(exclude_map[idx])) if idx in exclude_map else None

#             if img.ndim != 2 or mask.shape != img.shape:
#                 continue

#             # Apply chaining exclusion
#             if exclude_mask is not None:
#                 mask[exclude_mask > 0] = 0

#             # Generate patch-level coords
#             coords = generate_patches(mask, patch_size=256, stride=128, num_tns=30)

#             for label_tag, y, x in coords:
#                 # ---- CHANGED: skip bg-only patches entirely ----
#                 # y0 = max(0, y - 128)
#                 # x0 = max(0, x - 128)
#                 # patch_img = img[y0:y0+256, x0:x0+256]
#                 # patch_mask = mask[y0:y0+256, x0:x0+256]

#                 # inside load_image_mask_pairs loop, before making patch_img/patch_mask
#                 h, w = img.shape
#                 patch_size = 256
#                 half = patch_size // 2

#                 # center the 256×256 crop on (y, x) and clamp to image bounds
#                 y0 = min(max(0, y - half), max(0, h - patch_size))
#                 x0 = min(max(0, x - half), max(0, w - patch_size))

#                 patch_img = img[y0:y0+patch_size, x0:x0+patch_size]
#                 patch_mask = mask[y0:y0+patch_size, x0:x0+patch_size]




#                 # patch_mask = patch_mask.astype(np.int32)
#                 if patch_mask.max() == 0:
#                     # # Ensure mask is int32 and not empty
#                     # patch_mask = np.zeros_like(patch_mask, dtype=np.int32)
                    
#                     # do not add a sample for pure background
#                     continue

#                 # ensure dtype & contiguity
#                 patch_mask = np.ascontiguousarray(patch_mask.astype(np.int32))
#                 # >>> add a channel so shape is (1, H, W)
#                 patch_mask = patch_mask[None, :, :]   # (1, 256, 256)


#                 patch_img = np.ascontiguousarray(patch_img)

#                 images.append(patch_img)
#                 masks.append(patch_mask)
#                 patch_idx = len(images) - 1

#                 # tn_coords = []

#                 # # TN from patch selection
#                 # if label == 'bg':
#                 #     tn_coords.append((patch_idx, 128, 128))  # patch center


#                 # TNs from chaining mask (in patch coords), if any
#                 if exclude_mask is not None:
#                     patch_excl = exclude_mask[y0:y0+256, x0:x0+256]
#                     chaining_coords = np.argwhere(patch_excl > 0)
#                 #     for cy, cx in chaining_coords:
#                 #         tn_coords.append((patch_idx, cy, cx))
#                 # tns_all.append(tn_coords)
#                     if chaining_coords.size > 0:
#                         tns_all.append([(patch_idx, int(cy), int(cx)) for cy, cx in chaining_coords])
#                     else:
#                         tns_all.append([])  # keep structure aligned
#                 else:
#                     tns_all.append([])

#         except Exception as e:
#             print(f"❌ Skipping index {idx} due to error: {e}")
#             continue

#     return images, masks, tns_all

# # ========== MAIN ==========
# model = models.CellposeModel(gpu=True)
# all_images, all_masks, all_tns = [], [], []

# for triplet in folder_pairs:
#     if len(triplet) == 3:
#         img_dir, mask_dir, excl_dir = triplet
#     else:
#         img_dir, mask_dir = triplet
#         excl_dir = None

#     imgs, msks, tns = load_image_mask_pairs(img_dir, mask_dir, excl_dir)

#     base_idx = len(all_images)
#     all_images.extend(imgs)
#     all_masks.extend(msks)

# #     for i, coords in enumerate(tns):
# #         all_tns.extend((base_idx + img_idx, y, x) for img_idx, y, x in coords)

# # print(f"\n📊 Total images: {len(all_images)}")

#     # flatten and offset TNs
#     for coords in tns:
#         for (img_idx, y, x) in coords:
#             all_tns.append((base_idx + img_idx, y, x))

# print(f"\n📊 Total images (pre-filter): {len(all_images)}")

# # ---- CHANGED: Remove any leftover empties and remap TN indices ----
# keep = [i for i, m in enumerate(all_masks) if np.max(m) > 0]
# drop = set(range(len(all_masks))) - set(keep)
# if drop:
#     print(f"⚠️ Dropping {len(drop)} bg-only patches (no instances).")

# idx_map = {old: new for new, old in enumerate(keep)}

# all_images = [np.ascontiguousarray(all_images[i]) for i in keep]
# all_masks  = [np.ascontiguousarray(all_masks[i].astype(np.int32)) for i in keep]
# all_tns    = [(idx_map[i], y, x) for (i, y, x) in all_tns if i in idx_map]

# n_empty = sum(1 for m in all_masks if np.max(m) == 0)
# print(f"✅ kept={len(all_masks)} | empty={n_empty} (must be 0)")
# assert n_empty == 0, "Empty masks made it into training labels."

# # Split
# X_train, X_val, y_train, y_val = train_test_split(all_images, all_masks, test_size=0.2, random_state=42)

# # remap TNs into train/val partitions
# # train indices are [0, len(X_train)-1], val indices are [len(X_train), ...] in the compact arrays
# tn_train = []
# tn_val = []
# for (i, y, x) in all_tns:
#     if i < len(X_train):
#         tn_train.append((i, y, x))
#     else:
#         tn_val.append((i - len(X_train), y, x))

# # Cleanup
# torch.cuda.empty_cache()
# gc.collect()

# # Memory log (fixed units)
# p = psutil.Process(os.getpid())
# print(f"🧠 RAM: {p.memory_info().rss / 1024**3:.2f} GB | GPU: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

# # Train
# print(f"\n🚀 Training model: {model_name}")
# model_path, train_losses, test_losses = train.train_seg(
#     model.net,
#     train_data=X_train,
#     train_labels=y_train,
#     test_data=X_val,
#     test_labels=y_val,
#     tn_coords=tn_train,
#     test_tn_coords=tn_val,
#     batch_size=1,
#     n_epochs=70,
#     learning_rate=1e-5,
#     weight_decay=0.1,
#     nimg_per_epoch=30,
#     model_name=model_name
# )

# # Plot
# Path("loss_curves").mkdir(exist_ok=True)
# plt.plot(train_losses, label="Train Loss")
# plt.plot(test_losses, label="Val Loss")
# plt.title(f"Loss Curve: {model_name}")
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.grid(True)
# plt.legend()
# plt.savefig(f"loss_curves/loss_curve_{model_name}.png")
# plt.close()

# print(f"\n📈 Saved loss curve to: loss_curve_{model_name}.png")
# print(f"💾 Model saved at: {model_path}")