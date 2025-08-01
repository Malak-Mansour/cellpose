from cellpose import io, models, core, train
from skimage.measure import label
from pathlib import Path
import numpy as np
import re
import matplotlib.pyplot as plt
from skimage.transform import resize
from sklearn.model_selection import train_test_split

import os
from skimage.color import label2rgb
from skimage.io import imsave
from skimage.util import img_as_ubyte
from skimage.segmentation import find_boundaries


# ------------------ Helper Functions ------------------ #
def draw_square_on_image(img, y, x, size=7, color=(255, 0, 0)):
    half = size // 2
    y1 = max(0, y - half)
    y2 = min(img.shape[0], y + half + 1)
    x1 = max(0, x - half)
    x2 = min(img.shape[1], x + half + 1)
    img[y1:y2, x1:x2] = color

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

        '''
        # Debugging: Save resized mask and TN coordinates
        print(tn_coords)

        # Save GT overlay with TNs (for sanity check)
        debug_dir = Path("debug_outputs") / "final_overlay"
        debug_dir.mkdir(parents=True, exist_ok=True)

        if img.dtype == np.uint16 or img.max() > 1:
            norm_img = ((img - img.min()) / (img.max() - img.min())).astype(np.float32)
        else:
            norm_img = img.astype(np.float32)

        norm_img_rgb = np.stack([norm_img]*3, axis=-1) if norm_img.ndim == 2 else norm_img

        overlay_green = label2rgb(mask, image=norm_img_rgb, bg_label=0, alpha=0.4, colors=['lime'])

        overlay_uint8 = img_as_ubyte(np.clip(overlay_green, 0, 1))

        scale_y = mask.shape[0] / resized_mask.shape[0]
        scale_x = mask.shape[1] / resized_mask.shape[1]
        for y, x in tn_coords:
            y_orig = int(y * scale_y)
            x_orig = int(x * scale_x)
            # if 0 <= y_orig < overlay_uint8.shape[0] and 0 <= x_orig < overlay_uint8.shape[1]:
            #     overlay_uint8[y_orig, x_orig] = [255, 0, 0]  # Red TN marker
            draw_square_on_image(overlay_uint8, y_orig, x_orig, size=7, color=(255, 0, 0))

        imsave(debug_dir / f"overlay_img_mask_tn_{i}.png", overlay_uint8)
        '''

    return images, masks, tns_all

# ------------------ Paths ------------------ #


folder_pairs = [
    ("../organized_masks_data_7_7_2025/Jurkat/Dynamic/5. Jurkat_ON_OFF_10V_R1/10v_20s-40s_nd_089",
     "../organized_masks_data_7_7_2025/Jurkat/Dynamic/5. Jurkat_ON_OFF_10V_R1/Modified_masks"),

    ("../organized_masks_data_7_7_2025/Jurkat/Dynamic/5. Jurkat_ON_OFF_10V_R3/10v_20s-40s_nd_091",
     "../organized_masks_data_7_7_2025/Jurkat/Dynamic/5. Jurkat_ON_OFF_10V_R3/Modified_masks"),

    ("../organized_masks_data_7_7_2025/SKBR3/Dynamic/6. SKBR3_ON_OFF_10V_R1/L6",
     "../organized_masks_data_7_7_2025/SKBR3/Dynamic/6. SKBR3_ON_OFF_10V_R1/Modified_masks"),

    ("../organized_masks_data_7_7_2025/SKBR3/Dynamic/6. SKBR3_ON_OFF_10V_R2/L4",
     "../organized_masks_data_7_7_2025/SKBR3/Dynamic/6. SKBR3_ON_OFF_10V_R2/Modified_masks"),
    
    ("../organized_masks_data_7_7_2025/SKBR3/Dynamic/6. SKBR3_ON_OFF_10V_R3/L3_001",
     "../organized_masks_data_7_7_2025/SKBR3/Dynamic/6. SKBR3_ON_OFF_10V_R3/Modified_masks"),
    
    ("../organized_masks_data_7_7_2025/SKBR3/Static/3. SKBR3_Static_R1/L3",
     "../organized_masks_data_7_7_2025/SKBR3/Static/3. SKBR3_Static_R1/Modified_masks"),
    
    ("../organized_masks_data_7_7_2025/SKBR3/Static/3. SKBR3_Static_R2/L5",
     "../organized_masks_data_7_7_2025/SKBR3/Static/3. SKBR3_Static_R2/Modified_masks"),
    
    ("../organized_masks_data_7_7_2025/SKBR3/Static/3. SKBR3_Static_R3/L2",
     "../organized_masks_data_7_7_2025/SKBR3/Static/3. SKBR3_Static_R3/Modified_masks"),
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

    model_name = f"fine_model_test_eval_{stage_idx+1}"
    print(f"🚀 Training model: {model_name}")

    # model_path, train_losses, test_losses = train.train_seg(
    #     model.net,
    #     train_data=all_images,
    #     train_labels=all_masks,
    #     tn_coords=all_tns,
    #     batch_size=1,
    #     n_epochs=100,
    #     learning_rate=1e-5,
    #     weight_decay=0.1,
    #     nimg_per_epoch=min(60, len(all_images)),
    #     model_name=model_name
    # )

    # ADDING VALIDATION SPLIT
    # Split images, masks, and TNs into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(all_images, all_masks, test_size=0.2, random_state=42)

    # Adjust TNs for training and validation
    tn_train = [(i, y, x) for (i, y, x) in all_tns if i < len(X_train)]
    tn_val   = [(i - len(X_train), y, x) for (i, y, x) in all_tns if i >= len(X_train)]

    # Run training with validation data
    model_path, train_losses, test_losses = train.train_seg(
        model.net,
        train_data=X_train,
        train_labels=y_train,
        test_data=X_val,
        test_labels=y_val,
        tn_coords=tn_train,
        test_tn_coords=tn_val,
        batch_size=1,
        n_epochs=70,
        learning_rate=1e-5,
        weight_decay=0.1,
        nimg_per_epoch=min(60, len(X_train)),
        model_name=model_name
    )




    print(f"💾 Saved model from batch {stage_idx+1} to: {model_path}")

    # === Run inference on training images ===
    print("🔍 Running inference on training images...")
    preds, _, _ = model.eval(all_images, channels=[[0, 0]], batch_size=1)

    debug_overlay_dir = Path("debug_outputs") / f"overlay_predictions_batch_{stage_idx+1}"
    debug_overlay_dir.mkdir(parents=True, exist_ok=True)

    for i, img in enumerate(all_images):
        gt_mask = all_masks[i]
        pred_mask = preds[i]

        if img.dtype == np.uint16 or img.max() > 1:
            norm_img = ((img - img.min()) / (img.max() - img.min())).astype(np.float32)
        else:
            norm_img = img.astype(np.float32)

        norm_img_rgb = np.stack([norm_img]*3, axis=-1) if norm_img.ndim == 2 else norm_img

        overlay_pred = label2rgb(pred_mask, image=norm_img_rgb, bg_label=0, alpha=0.4, colors=['red'])
        overlay_gt = label2rgb(gt_mask, image=overlay_pred, bg_label=0, alpha=0.4, colors=['green'])

        overlay_uint8 = img_as_ubyte(np.clip(overlay_gt, 0, 1))
        imsave(debug_overlay_dir / f"debug_overlay_{i}.png", overlay_uint8)
        print(f"🖼️ Saved: debug_overlay_{i}.png")

    # === Plot loss ===
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
