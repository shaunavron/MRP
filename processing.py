import os
import re
import pandas as pd
import numpy as np
import nibabel as nib
import shutil
from scipy.ndimage import zoom
from sklearn.model_selection import train_test_split

DATA_DIR = "/Volumes/T9/"
OUTPUT_DIR_3D = "/Volumes/T9/processed/3d/"
OUTPUT_DIR_2D = "/Volumes/T9/processed/2d/"
TRAIN_DIR = "/Volumes/T9/processed/train/"
VAL_DIR = "/Volumes/T9/processed/val/"
TEST_DIR = "/Volumes/T9/processed/test/"

def resize_3d(mri):
    desired_width = 128
    desired_height = 128
    desired_depth = 128

    curr_width = mri.shape[0]   # Left-Right (L)
    curr_height = mri.shape[1]  # Anterior-Posterior (A)
    curr_depth = mri.shape[2]   # Superior-Inferior (S)

    width_factor = desired_width/curr_width
    height_factor = desired_height/curr_height
    depth_factor = desired_depth/curr_depth

    mri = zoom(mri, (width_factor, height_factor, depth_factor), order=1)
    return mri

def resize_2d(img):

    desired_width = 224
    desired_height = 224

    curr_width = img.shape[0]
    curr_height = img.shape[1]

    width_factor = desired_width / curr_width
    height_factor = desired_height / curr_height

    return zoom(img, (width_factor, height_factor), order=1)

def normalize(mri):
    vol = mri[mri > 0]
    return (mri - vol.mean()) / (vol.std() + 1e-6)

def copy_splits(df, dest_root):
        # 2D
    for _, row in df.iterrows():
        subject_id = row["ID"]

        if row["label"] == 0:
            class_label = "normal"
        else:
            class_label = "diseased"
        
        source_2d = OUTPUT_DIR_2D + class_label + "/" + subject_id
        destination_2d = dest_root + "2d/" + class_label + "/" + subject_id

        shutil.copytree(source_2d, destination_2d)

        # 3D
        source_3d = OUTPUT_DIR_3D + class_label + "/" + subject_id + ".npy"
        destination_3d = dest_root + "3d/" + class_label + "/" + subject_id + ".npy"
        
        shutil.copy2(source_3d, destination_3d)

def create_splits(df):
    df = df.dropna(subset=["CDR"]).copy()
    df["label"] = (df["CDR"] > 0).astype(int)
    
    train_df, temp_df = train_test_split(df, test_size=0.20, stratify=df["label"], random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df["label"], random_state=42)
    
    copy_splits(train_df, TRAIN_DIR)
    copy_splits(val_df, VAL_DIR)
    copy_splits(test_df ,TEST_DIR)


hdr_files = []

for root, dirs, files in os.walk(DATA_DIR):
    for file in files:
        if file.endswith("t88_gfc.hdr") and not file.startswith("._"):
            hdr_files.append(os.path.join(root, file))

labels = pd.read_excel("oasis1_cross-sectional-5708aa0a98d82080.xlsx")

cdr_lookup = dict(zip(labels["ID"], labels["CDR"]))

for mri_path in hdr_files:
    match = re.search(r"(OAS1_\d{4}_MR1)", mri_path)
    if not match:
        continue
    subject_id = match.group(1)
    cdr = cdr_lookup.get(subject_id)


    # 3D
    if pd.isna(cdr):
        continue
    if cdr == 0:
        temp_path = OUTPUT_DIR_3D + "normal/" + subject_id + ".npy"
    else:   # CDR = 0.5 (Very Mildly Demented), 1.0 (Mildly Demented) 2.0 (Moderately Demented) 3.0 (Extremely Demented)
        temp_path = OUTPUT_DIR_3D + "diseased/" + subject_id + ".npy"

    mri = nib.load(mri_path).get_fdata().astype(np.float32)
    mri = np.squeeze(mri)
    mri = normalize(mri)
    mri = resize_3d(mri)
    mri_3d = np.expand_dims(mri, axis=-1)
    np.save(temp_path, mri_3d)

    # 2D
    mid_slice = mri.shape[1] // 2 # access coronal slice
    if cdr == 0:
        temp_path = OUTPUT_DIR_2D + "normal/" + subject_id + "/"
    else:   # CDR = 0.5 (Very Mildly Demented), 1.0 (Mildly Demented) 2.0 (Moderately Demented) 3.0 (Extremely Demented)
        temp_path = OUTPUT_DIR_2D + "diseased/" + subject_id + "/"

    os.makedirs(temp_path, exist_ok=True)

    for i in range(mid_slice - 15, mid_slice + 16):
        slice = mri[:, i, :]
        slice = resize_2d(slice)
        np.save(temp_path + f"slice_{i:03d}.npy", slice)

create_splits(labels)

