'''
#TRAINING ON ALL IMAGES
python test.py \
  --model_name "/l/users/malak.mansour/DEP/cellpose/models/fine_model_all_without_chaining" \
  --img_dir "../flattened_dataset/all_images" \
  --img_pattern "img_*.tif" \
  --img_slice "-6:" \
  --overlay_folder "overlay_visualizations/all_images_before_chaining"


#CROSS DOMAIN GENERALIZATION
python test.py \
  --model_name "/l/users/malak.mansour/DEP/cellpose/models/fine_model_all_without_chaining" \
  --img_dir "../filtered_DEP_data/MDA_static_L4/Images" \
  --img_pattern "t*.tif" \
  --img_slice ":3" \
  --overlay_folder "overlay_visualizations/MDA"

  
#CHAINING
python test.py \
  --model_name "/l/users/malak.mansour/DEP/cellpose/models/fine_model_all_chaining" \
  --img_dir "../filtered_DEP_data/6. SKBR3_ON_OFF_10V_R1/Images/L6" \ #unseen images
  --img_pattern "t*.tif" \
  --img_slice ":3" \
  --overlay_folder "overlay_visualizations/Chaining"

python test.py \
  --model_name "/l/users/malak.mansour/DEP/cellpose/models/fine_model_all_chaining" \
  --img_dir "../filtered_DEP_data/3. SKBR3_Static_R1/Images/L3" \ #easy chaining example
  --img_pattern "t*.tif" \
  --img_slice ":3" \
  --overlay_folder "overlay_visualizations/Chaining"

python test.py \
  --model_name "/l/users/malak.mansour/DEP/cellpose/models/fine_model_all_chaining" \
  --img_dir "../filtered_DEP_data/5. Jurkat_ON_OFF_10V_R2/Images/10v_20s-40s_nd_090" \ #4 chained cells
  --img_pattern "t*.tif" \
  --img_slice ":3" \
  --overlay_folder "overlay_visualizations/Chaining"
'''
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from skimage.io import imread
from cellpose import models, core, io, utils
from tifffile import imsave
from cellpose.utils import masks_to_outlines
import argparse

# ------------------ Parse Args ------------------ #
def parse_args():
    parser = argparse.ArgumentParser(description="Cellpose Inference Script")
    parser.add_argument('--model_name', type=str, required=True, help='Path to custom trained model')
    parser.add_argument('--img_dir', type=str, required=True, help='Path to directory of test images')
    parser.add_argument('--img_slice', type=str, default=':', help='Slice string for image selection (e.g., ":2", "-6:", "1:5")')
    parser.add_argument('--img_pattern', type=str, default='t*.tif', help='Pattern to match image files (e.g., "t*.tif", "img_*.tif")')
    parser.add_argument('--overlay_folder', type=str, default='overlay_visualizations', help='Folder name to save overlays')
    return parser.parse_args()

args = parse_args()

# ------------------ Setup ------------------ #
io.logger_setup()
if not core.use_gpu():
    raise RuntimeError("❌ No GPU found. Enable GPU.")



# ------------------ Load Image Paths ------------------ #
image_dir = Path(args.img_dir)
all_images = sorted([f for f in image_dir.glob(args.img_pattern)])
image_files = eval(f"all_images[{args.img_slice}]")
print(f"✅ Found {len(image_files)} images")

# ------------------ Load Images ------------------ #
imgs = []
image_names = []
for f in image_files:
    img = imread(f).astype(np.float32)
    print(f"Image {f.name} shape: {img.shape}, dtype: {img.dtype}, {img.min()} - {img.max()}")

    if img.max() > 255:
        img = np.clip(img, 0, 65535)
        img = (img / 65535.0) * 255.0

    if img.ndim == 2:
        img = np.expand_dims(img, axis=0)
    elif img.ndim == 3 and img.shape[-1] == 3:
        img = img.transpose(2, 0, 1)

    imgs.append(img)
    image_names.append(f.name)

imgs = np.stack(imgs, axis=0)
print(f"🖼️ Input batch shape: {imgs.shape}")

# ------------------ Load Model ------------------ #
model = models.CellposeModel(gpu=True, pretrained_model=Path(args.model_name).expanduser())

# ------------------ Inference ------------------ #
masks_pred_all, flows_all, styles_all = [], [], []

for i, img in enumerate(imgs):
    masks_pred, flows, styles = model.eval(
        img,
        niter=1000,
        do_3D=False,
        channels=[0, 0],
        # flow_threshold=0.4,
        # cellprob_threshold=0.0,
    )
    masks_pred_all.append(masks_pred)
    flows_all.append(flows)
    styles_all.append(styles)

# ------------------ Save Predicted Masks ------------------ #
output_dir = Path("predicted_masks")
output_dir.mkdir(exist_ok=True)

for i, mask in enumerate(masks_pred_all):
    n_cells = mask.max()
    print(f"{image_names[i]} → {n_cells} predicted cells")
    imsave(output_dir / f"{image_names[i].replace('.tif', '')}_seg.tif", mask.astype(np.uint16))

print(f"✅ Saved {len(masks_pred_all)} predicted masks to: {output_dir.resolve()}")

# ------------------ Save Overlay Visualizations ------------------ #
viz_dir = Path(args.overlay_folder)
viz_dir.mkdir(exist_ok=True)

for i, (img, mask) in enumerate(zip(imgs, masks_pred_all)):
    img_vis = img[0]

    # Contrast stretch
    p2, p98 = np.percentile(img_vis, (2, 98))
    img_vis = np.clip((img_vis - p2) / (p98 - p2 + 1e-5), 0, 1)
    img_vis = (img_vis * 100).astype(np.uint8)

    outlines = masks_to_outlines(mask)
    overlay = np.stack([img_vis.copy()] * 3, axis=-1)
    overlay[outlines, :] = [255, 0, 0]

    out_path = viz_dir / f"{image_names[i].replace('.tif', '')}_overlay.png"
    plt.imsave(out_path, overlay)
    print(f"🖼️ Saved overlay to: {out_path}")