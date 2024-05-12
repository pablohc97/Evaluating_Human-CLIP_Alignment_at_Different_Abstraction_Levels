# %%
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from PIL import Image

import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
device = "cuda"

from open_clip import create_model_from_pretrained, get_tokenizer # works on open-clip-torch>=2.23.0, timm>=0.9.8

model, preprocess = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
model.eval()
tokenizer = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')

night_path = '/lustre/ific.uv.es/ml/uv075/Databases/HighLevel/Nights/nights/'
data_night = pd.read_csv(night_path+'data.csv')

distances1_clip = np.zeros((20019,512))
distances2_clip = np.zeros((20019,512))

for i, row in tqdm(data_night.iterrows(), total=20019):
    img = Image.open(night_path + row.ref_path)
    dist_img_1 = Image.open(night_path + row.left_path)
    dist_img_2 = Image.open(night_path+ row.right_path)

    img_inputs = preprocess(img)[None]#.unsqueeze(0).to(device)
    dist_img_1_inputs = preprocess(dist_img_1)[None]#.unsqueeze(0).to(device)
    dist_img_2_inputs = preprocess(dist_img_2)[None]#.unsqueeze(0).to(device)

    feature_img = model.visual(img_inputs)
    feature_dist_1_img = model.visual(dist_img_1_inputs)
    feature_dist_2_img = model.visual(dist_img_2_inputs)

    d_1 = np.mean((feature_img.detach().numpy() - feature_dist_1_img.detach().numpy())**2, axis=0)
    d_2 = np.mean((feature_img.detach().numpy() - feature_dist_2_img.detach().numpy())**2, axis=0)

    distances1_clip[i,:] = d_1
    distances2_clip[i,:] = d_2


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
print(f"BiomedCLIP base16 NightsPerceptualDataset accuracy projection euclidean norm = {100*correct/total:.3f}")
