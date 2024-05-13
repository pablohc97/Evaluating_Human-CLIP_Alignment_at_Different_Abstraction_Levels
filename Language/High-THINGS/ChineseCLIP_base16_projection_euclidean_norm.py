# %%
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import glob
from PIL import Image

import os
device = "cuda"

# %%
from transformers import AutoProcessor, ChineseCLIPModel

model = ChineseCLIPModel.from_pretrained("OFA-Sys/chinese-clip-vit-base-patch16").to(device)
processor = AutoProcessor.from_pretrained("OFA-Sys/chinese-clip-vit-base-patch16")

# %%
data_things = pd.read_csv('/lustre/ific.uv.es/ml/uv075/Databases/HighLevel/THINGS/triplets/triplet_dataset/testset1_paths.csv')
class_numers_things = pd.read_csv('/lustre/ific.uv.es/ml/uv075/Databases/HighLevel/THINGS/triplets/triplet_dataset/class_name_number.csv')
things_path = '/lustre/ific.uv.es/ml/uv075/Databases/HighLevel/THINGS'
images_things_path = '/lustre/ific.uv.es/ml/uv075/Databases/HighLevel/THINGS/THINGS/Images/'

# %%
images = np.zeros((len(class_numers_things),1,512))
for i, row in tqdm(class_numers_things.iterrows(), total=len(class_numers_things)):
    folder_path = images_things_path + row.ClassID
    img_path = sorted(glob.glob(folder_path + '/*'))[0]
    img = Image.open(img_path)
    input_img = processor(images=img, text=['杰尼龟'], padding=True, return_tensors="pt").to(device)
    features_img = model(**input_img).image_embeds
    images[i,:] = features_img.cpu().detach().numpy()
    
print(images.shape)

distances_12 = np.zeros((len(data_things),512))
distances_13 = np.zeros((len(data_things),512))
distances_23 = np.zeros((len(data_things),512))

for i, row in tqdm(data_things.iterrows(), total=len(data_things)):
    img_1 = images[row.Im1]
    img_2 = images[row.Im2]
    img_3 = images[row.Im3]

    d_12 = np.mean((img_1 - img_2)**2, axis=0)
    d_13 = np.mean((img_1 - img_3)**2, axis=0)
    d_23 = np.mean((img_2 - img_3)**2, axis=0)

    distances_12[i,:] = d_12
    distances_13[i,:] = d_13
    distances_23[i,:] = d_23


distances_12_means = distances_12.mean(axis=0)
distances_13_means = distances_13.mean(axis=0)
distances_23_means = distances_23.mean(axis=0)

distances_means = np.array([distances_12_means, distances_13_means, distances_23_means]).mean(axis=0)

distances_12_norm = np.zeros((len(data_things),512))
distances_13_norm = np.zeros((len(data_things),512))
distances_23_norm = np.zeros((len(data_things),512))

for i in range(distances_12_norm.shape[-1]):
    distances_12_norm[:,i] = distances_12[:,i] / distances_means[i]
    distances_13_norm[:,i] = distances_13[:,i] / distances_means[i]
    distances_23_norm[:,i] = distances_23[:,i] / distances_means[i]

distances12_clip_norm_fin = np.sqrt(distances_12_norm.mean(axis=1))
distances13_clip_norm_fin = np.sqrt(distances_13_norm.mean(axis=1))
distances23_clip_norm_fin = np.sqrt(distances_23_norm.mean(axis=1))


# %%
correct = 0
total = 0

for i, row in tqdm(data_things.iterrows(), total=len(data_things)):

    total += 1

    if (distances12_clip_norm_fin[i] < distances13_clip_norm_fin[i]) and (distances12_clip_norm_fin[i] < distances23_clip_norm_fin[i]):
        correct += 1
    else:
        pass

# %%
print(f'Total images {total}')
print(f"ChineseCLIP base16 THINGS-odd-one-out accuracy projection euclidean norm = {100*correct/total:.3f}")


