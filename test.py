# # pip install git+https://www.github.com/mouseland/cellpose.git

# import numpy as np
# from cellpose import models, core, io, plot
# from pathlib import Path
# from tqdm import trange
# import matplotlib.pyplot as plt

# io.logger_setup() # run this to get printing of progress

# #Check if colab notebook instance has GPU access
# if core.use_gpu()==False:
#   raise ImportError("No GPU access, change your runtime")

# model = models.CellposeModel(gpu=True)





# import os
# import numpy as np
# import matplotlib.pyplot as plt
# from skimage.io import imread
# from cellpose import models, transforms, utils

# # Load Cellpose model (adjust model_type if needed)
# model = models.CellposeModel(gpu=True, model_type='cyto')  # ✅ New

# # Path to your folder of .tif images
# image_folder = 'images'  
# image_files = sorted([f for f in os.listdir(image_folder) if f.endswith('.tif')])[:2]
# print(image_files)

# # Load a few images (e.g. 6 max for nice visualization)
# imgs = []
# image_names = []
# for fname in sorted(image_files):
#     img = imread(os.path.join(image_folder, fname))
#     if img.ndim == 2:
#         img = np.expand_dims(img, axis=0)  # grayscale: (1, H, W)
#     elif img.ndim == 3 and img.shape[-1] == 3:
#         img = img.transpose(2, 0, 1)        # RGB: (3, H, W)
#     imgs.append(img)
#     image_names.append(fname)

# imgs = np.stack(imgs, axis=0)  # shape: (N, C, H, W)
# print(imgs.shape)





# # Prepare output lists
# masks_pred_all, flows_all, styles_all = [], [], []


# for img in imgs:
#   # Inference using more iterations (e.g., for bacterial images)
#   masks_pred, flows, styles = model.eval(
#       img,
#       niter=1000,
#       do_3D=False,      # ✅ force 2D
#       channels=[0, 0]   # ✅ grayscale; adjust to [0,1] if using 2-channel input
#   )
#   masks_pred_all.append(masks_pred)
#   flows_all.append(flows)
#   styles_all.append(styles)





# # Titles for visualization
# titles = image_names


# # Plotting predictions
# plt.figure(figsize=(14, 6))
# for iex in range(len(imgs)):
#     img = imgs[iex].copy()
#     img = np.clip(transforms.normalize_img(img, axis=0), 0, 1)

#     ax = plt.subplot(2, 3, iex + 1)
#     if img.shape[0] == 1:
#         ax.imshow(img[0], cmap="gray")
#     else:
#         ax.imshow(img.transpose(1, 2, 0))

#     # Draw predicted mask outlines in yellow
#     outlines_pred = utils.outlines_list(masks_pred_all[iex])
#     for o in outlines_pred:
#         plt.plot(o[:, 0], o[:, 1], color=[1, 1, 0.3], lw=0.75, ls="--")

#     ax.set_title(titles[iex])
#     ax.axis('off')


# plt.tight_layout()
# plt.show()


import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from skimage.io import imread
from cellpose import models, core, io, transforms, utils

io.logger_setup()
if not core.use_gpu():
    raise RuntimeError("❌ No GPU found. Enable GPU.")

# Path to your images
image_dir = Path("../organized_masks_data_7_7_2025/Jurkat/Dynamic/5. Jurkat_ON_OFF_10V_R2/10v_20s-40s_nd_090")
image_files = sorted([f for f in image_dir.glob("t*.tif")])
print(f"✅ Found {len(image_files)} images")


# Load images
imgs = []
image_names = []
for f in image_files:


    img = imread(f)
    print(f"Image {f.name} shape: {img.shape}, dtype: {img.dtype}, {img.min()} - {img.max()}")

    # Convert to float32 but DO NOT normalize
    img = img.astype(np.float32)

    # # Normalize to 0-1 if image is 16-bit
    # img = imread(f).astype(np.float32)

    # print(f"Image {f.name} shape: {img.shape}, dtype: {img.dtype}, {img.min()} - {img.max()}")
    # if img.max() > 255:
    #     img = np.clip(img, 0, 65535)  # ensure range
    #     img = (img / 65535.0) * 255.0


    if img.ndim == 2:
        img = np.expand_dims(img, axis=0)  # shape: (1, H, W)
    elif img.ndim == 3 and img.shape[-1] == 3:
        img = img.transpose(2, 0, 1)        # RGB to (C, H, W)
    imgs.append(img)
    image_names.append(f.name)

imgs = np.stack(imgs, axis=0)
print(f"🖼️ Input batch shape: {imgs.shape}")  # (N, C, H, W)

# ------------------ Load Custom Model ------------------
# Update this path with the actual path where your model was saved
custom_model_path = "~/.cellpose/models/fine_model_batch_8"  

# model = models.CellposeModel(gpu=True, model_type='cyto')
model = models.CellposeModel(gpu=True, pretrained_model=Path(custom_model_path).expanduser())

# ------------------ Run Inference ------------------
masks_pred_all, flows_all, styles_all = [], [], []

for img in imgs:
    masks_pred, flows, styles = model.eval(
        img,
        niter=1000,
        do_3D=False,
        channels=[0, 0],  # grayscale input
        flow_threshold=0.9,         # default is 0.4
        cellprob_threshold=0.9      # default is 0.0
    )
    masks_pred_all.append(masks_pred)
    flows_all.append(flows)
    styles_all.append(styles)


from tifffile import imsave  # optional if saving as .tif

# Output directory for saving predicted masks
output_dir = Path("predicted_masks")
output_dir.mkdir(exist_ok=True)

for i, mask in enumerate(masks_pred_all):
    n_cells = mask.max()  # Cellpose labels cells with integers: 1, 2, 3, ...
    print(f"{image_names[i]} → {n_cells} predicted cells")

    imsave(output_dir / f"{image_names[i].replace('.tif', '')}_seg.tif", mask.astype(np.uint16))

print(f"✅ Saved {len(masks_pred_all)} predicted masks to: {output_dir.resolve()}")


from cellpose.utils import outlines_list, masks_to_outlines

viz_dir = Path("overlay_visualizations")
viz_dir.mkdir(exist_ok=True)

for i, (img, mask) in enumerate(zip(imgs, masks_pred_all)):
    img_vis = img[0]  # extract (H, W) from (1, H, W)

    # Normalize for display
    img_vis = np.clip(img_vis, 0, 255).astype(np.uint8)

    # Get outlines
    outlines = masks_to_outlines(mask)

    # Overlay outlines in red
    overlay = np.stack([img_vis.copy()] * 3, axis=-1)  # grayscale to RGB
    overlay[outlines, :] = [255, 0, 0]  # red lines

    # Save visualization
    out_path = viz_dir / f"{image_names[i].replace('.tif', '')}_overlay.png"
    plt.imsave(out_path, overlay)
    print(f"🖼️ Saved overlay to: {out_path}")
