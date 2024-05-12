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

from open_clip import create_model_from_pretrained, get_tokenizer # works on open-clip-torch>=2.23.0, timm>=0.9.8

model, preprocess = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
model.eval()
tokenizer = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')


tid_path_2013 = '/lustre/ific.uv.es/ml/uv075/Databases/IQA/TID/TID2013'
data_tid_2013 = pd.read_csv(tid_path_2013 + '/image_pairs_mos.csv', index_col = 0)

device = "cpu"

distances_2013_biomedclip = np.zeros((3000,512))

for i, row in tqdm(data_tid_2013.iterrows(), total=3000):
    img = Image.open(tid_path_2013 + '/reference_images/' + row.Reference)
    dist_img = Image.open(tid_path_2013 + '/distorted_images/' + row.Distorted)

    img_inputs = preprocess(img)[None]#.unsqueeze(0).to(device)
    dist_img_inputs = preprocess(dist_img)[None]#.unsqueeze(0).to(device)

    feature_img = model.visual(img_inputs)
    feature_dist_img = model.visual(dist_img_inputs)

    d = np.mean((feature_img.detach().numpy() - feature_dist_img.detach().numpy())**2, axis=(0))
    distances_2013_biomedclip[i,:] = d


distances_2013_biomedclip_means = distances_2013_biomedclip.mean(axis=0)

distances_2013_biomedclip_norm = np.zeros((3000,512))

for i in range(512):
    distances_2013_biomedclip_norm[:,i] = distances_2013_biomedclip[:,i] / distances_2013_biomedclip_means[i]


print(f'BioMedCLIP base16 TID-2013 spearman corr projection euclidean norm= {-spearmanr(np.sqrt(distances_2013_biomedclip_norm.mean(axis=1)),data_tid_2013.MOS)[0]:.3f}')