# %%
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from PIL import Image
from scipy.stats import pearsonr, spearmanr

import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
device = "cuda"

from numpy import dot
from numpy.linalg import norm

def cos_sim(a,b):
    return dot(a, b)/(norm(a)*norm(b))

from transformers import AutoProcessor, CLIPVisionModel
model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch16").to(device)     # if model = base -> layers = 13; if model = large -> layers = 25
processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch16")


tid_path_2013 = '/lustre/ific.uv.es/ml/uv075/Databases/IQA/TID/TID2013'
data_tid_2013 = pd.read_csv(tid_path_2013 + '/image_pairs_mos.csv', index_col = 0)


sim_2013_clip = np.zeros((3000,13))

for i, row in tqdm(data_tid_2013.iterrows(), total=3000):
    img = Image.open(tid_path_2013 + '/reference_images/' + row.Reference)
    dist_img = Image.open(tid_path_2013 + '/distorted_images/' + row.Distorted)

    img_inputs = processor(images=img, return_tensors="pt").to(device)
    dist_img_inputs = processor(images=dist_img, return_tensors="pt").to(device)

    with torch.no_grad():
        features_img = model(**img_inputs, output_hidden_states=True).hidden_states
        features_dist_img = model(**dist_img_inputs, output_hidden_states=True).hidden_states
    
    for j in range(len(features_img)):
        sim = cos_sim(features_img[j].cpu().numpy().flatten(), features_dist_img[j].cpu().numpy().flatten())
        sim_2013_clip[i,j] = sim

for k in range(len(features_img)):
    data_tid_2013[f'Sim_CLIP_{k}'] = sim_2013_clip[:,k]

for k in range(len(features_img)):
    pearson = pearsonr(data_tid_2013.MOS, data_tid_2013[f'Sim_CLIP_{k}'])[0]
    spearman = spearmanr(data_tid_2013.MOS, data_tid_2013[f'Sim_CLIP_{k}'])[0]
    print(f"CLIP base16 corr in TID-2013 layer {k} cosine: Pearson = {pearson:.3f}     Spearman = {spearman:.3f}")