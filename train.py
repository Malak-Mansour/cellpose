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

#     handler = RotatingFileHandler(log_file, maxBytes=5 * 1024 * 1024, backupCount=3)
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
















from cellpose import io, models, core, train
from skimage.measure import label
from pathlib import Path
import numpy as np
import re
import matplotlib.pyplot as plt
from skimage.transform import resize

# ------------------ Helper Functions ------------------ #

def extract_index(filename):
    match = re.search(r"(\d+)", filename.stem)
    return int(match.group(1)) if match else None

def bin_to_labels(mask):
    threshold = np.max(mask) * 0.5
    binary = mask > threshold
    labeled = label(binary.astype(np.uint8), connectivity=1)
    return labeled

def sample_true_negatives_from_resized_mask(resized_mask, num_samples=30):
    tn_coords = []
    bg_coords = np.argwhere(resized_mask == 0)
    if len(bg_coords) > 0:
        sampled = bg_coords[np.random.choice(len(bg_coords), min(num_samples, len(bg_coords)), replace=False)]
        for yx in sampled:
            tn_coords.append((yx[0], yx[1]))
    return tn_coords

def load_image_mask_pairs(image_dir, mask_dir, target_shape=(256, 256), num_tns=30):
    image_files = sorted(image_dir.glob("t*.tif"))
    mask_files = sorted(mask_dir.glob("masks*.tif"))

    image_map = {extract_index(f): f for f in image_files}
    mask_map = {extract_index(f): f for f in mask_files}

    common_indices = sorted(set(image_map.keys()) & set(mask_map.keys()))
    print(f"📂 {image_dir.name} → {len(common_indices)} matched pairs")

    images, masks, tns_all = [], [], []

    for i in common_indices:
        img = io.imread(image_map[i])
        mask = bin_to_labels(io.imread(mask_map[i]))

        resized_mask = resize(mask.astype(np.uint8), target_shape, order=0, preserve_range=True, anti_aliasing=False).astype(np.uint8)
        tn_coords = sample_true_negatives_from_resized_mask(resized_mask, num_tns)

        images.append(img)
        masks.append(mask)
        tns_all.append(tn_coords)

    return images, masks, tns_all

# ------------------ Paths ------------------ #

folder_pairs = [
    ("../organized_masks_data_7_7_2025/Jurkat/Dynamic/5. Jurkat_ON_OFF_10V_R1/10v_20s-40s_nd_089",
    "../organized_masks_data_7_7_2025/Jurkat/Dynamic/5. Jurkat_ON_OFF_10V_R1/Modified_masks")
]

batch_size = 1
batches = [folder_pairs[i:i+batch_size] for i in range(0, len(folder_pairs), batch_size)]

model_path = None

for stage_idx, folder_batch in enumerate(batches):
    print(f"\n🧪 Training batch {stage_idx+1}/{len(batches)}")

    all_images, all_masks, all_tns = [], [], []

    for img_path, mask_path in folder_batch:
        images, masks, tn_lists = load_image_mask_pairs(Path(img_path), Path(mask_path))
        base_idx = len(all_images)
        all_images.extend(images)
        all_masks.extend(masks)
        for i, tn_coords in enumerate(tn_lists):
            for y, x in tn_coords:
                all_tns.append((base_idx + i, y, x))

    print(f"✅ Batch {stage_idx+1}: {len(all_images)} images, {len(all_masks)} masks")

    if model_path:
        model = models.CellposeModel(gpu=True, pretrained_model=model_path)
    else:
        model = models.CellposeModel(gpu=True)

    model_name = f"fine_model_batch_test"
    print(f"🚀 Training model: {model_name}")

    model_path, train_losses, test_losses = train.train_seg(
        model.net,
        train_data=all_images,
        train_labels=all_masks,
        tn_coords=all_tns,
        batch_size=1,
        n_epochs=100,
        learning_rate=1e-5,
        weight_decay=0.1,
        nimg_per_epoch=min(60, len(all_images)),
        model_name=model_name
    )

    print(f"💾 Saved model from batch {stage_idx+1} to: {model_path}")

    plt.figure(figsize=(10,5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(test_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Loss Curve for {model_name}")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"loss_curve_{model_name}.png")
    print(f"📈 Saved loss curve as: loss_curve_{model_name}.png")

print(f"\n✅ All batches complete. Final model saved at: {model_path}")
