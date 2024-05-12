# %%
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from PIL import Image

import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
device = "cuda"

from transformers import AutoProcessor, CLIPVisionModelWithProjection
model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch16").to(device)
processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch16")


night_path = '/lustre/ific.uv.es/ml/uv075/Databases/HighLevel/Nights/nights/'
data_night = pd.read_csv(night_path+'data.csv')


distances1_clip = np.zeros((20019,512))
distances2_clip = np.zeros((20019,512))

for i, row in tqdm(data_night.iterrows(), total=20019):
    img = Image.open(night_path + row.ref_path)
    dist_img_1 = Image.open(night_path + row.left_path)
    dist_img_2 = Image.open(night_path+ row.right_path)

    input_img = processor(images=img, return_tensors="pt").to(device)
    input_dist_img_1 = processor(images=dist_img_1, return_tensors="pt").to(device)
    input_dist_img_2 = processor(images=dist_img_2, return_tensors="pt").to(device)

    with torch.no_grad():
        features_img = model(**input_img).image_embeds
        features_dist_img_1 = model(**input_dist_img_1).image_embeds
        features_dist_img_2 = model(**input_dist_img_2).image_embeds
    
    d1 = np.mean((features_img.cpu().numpy() - features_dist_img_1.cpu().numpy())**2, axis=0)
    d2 = np.mean((features_img.cpu().numpy() - features_dist_img_2.cpu().numpy())**2, axis=0)
    distances1_clip[i,:] = d1
    distances2_clip[i,:] = d2

distances1_clip_means = distances1_clip.mean(axis=0)
distances2_clip_means = distances2_clip.mean(axis=0)

distances_means = np.array([distances1_clip_means, distances2_clip_means]).mean(axis=0)

distances1_clip_norm = np.zeros((20019,512))
distances2_clip_norm = np.zeros((20019,512))

for i in range(distances2_clip_norm.shape[-1]):
    distances1_clip_norm[:,i] = distances1_clip[:,i] / distances_means[i]  
    distances2_clip_norm[:,i] = distances2_clip[:,i] / distances_means[i]  


distances1_clip_norm_fin = np.sqrt(distances1_clip_norm.mean(axis=1))
distances2_clip_norm_fin = np.sqrt(distances2_clip_norm.mean(axis=1))


correct = 0
total = 0

for i, row in tqdm(data_night.iterrows(), total=20019):
    total += 1

    if (distances1_clip_norm_fin[i] > distances2_clip_norm_fin[i]) and (row.right_vote == 1):
        correct += 1
    elif (distances1_clip_norm_fin[i] < distances2_clip_norm_fin[i]) and (row.left_vote == 1):
        correct += 1
    else:
        pass


print(f'Total images {total}')
print(f"CLIP base16 NightsPerceptualDataset accuracy projection euclidean norm = {100*correct/total:.3f}")