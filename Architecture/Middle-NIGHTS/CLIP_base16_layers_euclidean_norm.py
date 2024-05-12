# %%
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from PIL import Image

import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
device = "cuda"

from transformers import AutoProcessor, CLIPVisionModel
model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch16").to(device)     # if model = base -> layers = 13; if model = large -> layers = 25
processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch16")

night_path = '/lustre/ific.uv.es/ml/uv075/Databases/HighLevel/Nights/nights/'
data_night = pd.read_csv(night_path+'data.csv')


d1 = np.zeros((13,20019,768))
d2 = np.zeros((13,20019,768))

for i, row in tqdm(data_night.iterrows(), total=20019):
    img = Image.open(night_path + row.ref_path)
    dist_img_1 = Image.open(night_path + row.left_path)
    dist_img_2 = Image.open(night_path+ row.right_path)

    input_img = processor(images=img, return_tensors="pt").to(device)
    input_dist_img_1 = processor(images=dist_img_1, return_tensors="pt").to(device)
    input_dist_img_2 = processor(images=dist_img_2, return_tensors="pt").to(device)

    with torch.no_grad():
        features_img = model(**input_img, output_hidden_states=True).hidden_states
        features_dist_img_1 = model(**input_dist_img_1, output_hidden_states=True).hidden_states
        features_dist_img_2 = model(**input_dist_img_2, output_hidden_states=True).hidden_states
    
    for j in range(len(features_img)):
        d_1 = np.mean((features_img[j].cpu().numpy() - features_dist_img_1[j].cpu().numpy())**2, axis=(0,1))
        d_2 = np.mean((features_img[j].cpu().numpy() - features_dist_img_2[j].cpu().numpy())**2, axis=(0,1))

        d1[j,i,:] = d_1
        d2[j,i,:] = d_2


d1_means = d1.mean(axis=1)
d2_means = d2.mean(axis=1)

d_means = np.array([d1_means, d2_means]).mean(axis=0)

d1_norm = np.zeros((13,20019,768))
d2_norm = np.zeros((13,20019,768))

for i in range(len(features_img)):
    for j in range(d1_norm.shape[-1]):
        d1_norm[i,:,j] = d1[i,:,j] / d_means[i,j] 
        d2_norm[i,:,j] = d2[i,:,j] / d_means[i,j]  

d1_norm_fin = np.sqrt(d1_norm.mean(axis=2))
d2_norm_fin = np.sqrt(d2_norm.mean(axis=2))


correct = np.zeros(13)
total = 0


for i, row in tqdm(data_night.iterrows(), total=20019):
    total += 1

    for j in range(len(features_img)):
        if (d1_norm_fin[j,i] > d2_norm_fin[j,i]) and (row.right_vote == 1):
            correct[j] += 1
        elif (d1_norm_fin[j,i] < d2_norm_fin[j,i]) and (row.left_vote == 1):
            correct[j] += 1
        else:
            pass


print(f'Total images {total}')
for k in range(len(correct)):
    print(f"CLIP base16 NightsPerceptualDataset accuracy layer {k} = {100*correct[k]/total:.3f}")