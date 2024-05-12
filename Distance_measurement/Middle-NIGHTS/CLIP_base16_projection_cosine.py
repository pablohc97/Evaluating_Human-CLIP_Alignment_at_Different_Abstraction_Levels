# %%
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from PIL import Image

import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
device = "cuda"

from numpy import dot
from numpy.linalg import norm

def cos_sim(a,b):
    return dot(a, b)/(norm(a)*norm(b))


from transformers import AutoProcessor, CLIPVisionModelWithProjection
model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch16").to(device)   
processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch16")


night_path = '/lustre/ific.uv.es/ml/uv075/Databases/HighLevel/Nights/nights/'
data_night = pd.read_csv(night_path+'data.csv')


correct = 0
total = 0

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

    total += 1

    sim_1 = cos_sim(features_img.cpu().numpy().flatten(), features_dist_img_1.cpu().numpy().flatten())
    sim_2 = cos_sim(features_img.cpu().numpy().flatten(), features_dist_img_2.cpu().numpy().flatten())

    if (sim_1 < sim_2) and (row.right_vote == 1):
        correct += 1
    elif (sim_1 > sim_2) and (row.left_vote == 1):
        correct += 1
    else:
        pass


print(f'Total images {total}')
print(f"CLIP base16 NightsPerceptualDataset accuracy projection cosine = {100*correct/total:.3f}")