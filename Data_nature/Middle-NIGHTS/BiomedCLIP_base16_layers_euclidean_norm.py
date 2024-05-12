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

d1 = np.zeros((13,20019,768))
d2 = np.zeros((13,20019,768))

for i, row in tqdm(data_night.iterrows(), total=20019):
    img = Image.open(night_path + row.ref_path)
    dist_img_1 = Image.open(night_path + row.left_path)
    dist_img_2 = Image.open(night_path+ row.right_path)

    img_inputs = preprocess(img)[None]#.unsqueeze(0).to(device)
    dist_img_1_inputs = preprocess(dist_img_1)[None]#.unsqueeze(0).to(device)
    dist_img_2_inputs = preprocess(dist_img_2)[None]#.unsqueeze(0).to(device)

    feature_img = list(model.visual.trunk.children())[0](img_inputs)
    feature_dist_1_img = list(model.visual.trunk.children())[0](dist_img_1_inputs)
    feature_dist_2_img = list(model.visual.trunk.children())[0](dist_img_2_inputs)

    d_1 = np.mean((feature_img.detach().numpy() - feature_dist_1_img.detach().numpy())**2, axis=(0,1))
    d_2 = np.mean((feature_img.detach().numpy() - feature_dist_2_img.detach().numpy())**2, axis=(0,1))

    d1[0,i,:] = d_1
    d2[0,i,:] = d_2


    with torch.no_grad():
        for j, block in enumerate(list(model.visual.trunk.children())[4].children()):
            feature_img = block(feature_img)
            feature_dist_1_img = block(feature_dist_1_img)
            feature_dist_2_img = block(feature_dist_2_img)

            d_1 = np.mean((feature_img.detach().numpy() - feature_dist_1_img.detach().numpy())**2, axis=(0,1))
            d_2 = np.mean((feature_img.detach().numpy() - feature_dist_2_img.detach().numpy())**2, axis=(0,1))
            
            d1[j+1,i,:] = d_1
            d2[j+1,i,:] = d_2

d1_means = d1.mean(axis=1)
d2_means = d2.mean(axis=1)

d_means = np.array([d1_means, d2_means]).mean(axis=0)

d1_norm = np.zeros((13,20019,768))
d2_norm = np.zeros((13,20019,768))

for i in range(13):
    for j in range(d1_norm.shape[-1]):
        d1_norm[i,:,j] = d1[i,:,j] / d_means[i,j] 
        d2_norm[i,:,j] = d2[i,:,j] / d_means[i,j] 

d1_norm_fin = np.sqrt(d1_norm.mean(axis=2))
d2_norm_fin = np.sqrt(d2_norm.mean(axis=2))



correct = np.zeros(13)
total = 0

for i, row in tqdm(data_night.iterrows(), total=20019):
    total += 1

    for j in range(13):
        if (d1_norm_fin[j,i] > d2_norm_fin[j,i]) and (row.right_vote == 1):
            correct[j] += 1
        elif (d1_norm_fin[j,i] < d2_norm_fin[j,i]) and (row.left_vote == 1):
            correct[j] += 1
        else:
            pass


print(f'Total images {total}')
for k in range(len(correct)):
    print(f"BIOMEDCLIP base16 NightsPerceptualDataset accuracy layer euclidean norm {k} = {100*correct[k]/total:.3f}")