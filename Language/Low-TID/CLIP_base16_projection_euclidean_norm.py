# %%
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from PIL import Image
from scipy.stats import spearmanr

import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
device = "cuda"

from transformers import AutoProcessor, CLIPVisionModelWithProjection
model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch16").to(device)
processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch16")


tid_path_2013 = '/lustre/ific.uv.es/ml/uv075/Databases/IQA/TID/TID2013'
data_tid_2013 = pd.read_csv(tid_path_2013 + '/image_pairs_mos.csv', index_col = 0)


distances_2013_clip = np.zeros((3000,512))

for i, row in tqdm(data_tid_2013.iterrows(), total=3000):
    img = Image.open(tid_path_2013 + '/reference_images/' + row.Reference)
    dist_img = Image.open(tid_path_2013 + '/distorted_images/' + row.Distorted)

    img_inputs = processor(images=img, return_tensors="pt").to(device)
    dist_img_inputs = processor(images=dist_img, return_tensors="pt").to(device)

    with torch.no_grad():
        features_img = model(**img_inputs).image_embeds
        features_dist_img = model(**dist_img_inputs).image_embeds
    
    d = np.mean((features_img.cpu().numpy() - features_dist_img.cpu().numpy())**2, axis=0)
    distances_2013_clip[i,:] = d

distances_2013_clip_means = distances_2013_clip.mean(axis=0)

distances_2013_clip_norm = np.zeros((3000,512))

for i in range(distances_2013_clip_norm.shape[-1]):
    distances_2013_clip_norm[:,i] = distances_2013_clip[:,i] / distances_2013_clip_means[i]


print(f'CLIP base16 TID-2013 spearman corr projection euclidean norm = {-spearmanr(np.sqrt(distances_2013_clip_norm.mean(axis=1)),data_tid_2013.MOS)[0]:.3f}')