# python train.py


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