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

from transformers import AutoProcessor, CLIPVisionModel
model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch16").to(device)     # if model = base -> layers = 13; if model = large -> layers = 25
processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch16")


night_path = '/lustre/ific.uv.es/ml/uv075/Databases/HighLevel/Nights/nights/'
data_night = pd.read_csv(night_path+'data.csv')


correct = np.zeros(13)
total = 0

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

    total += 1

    for j in range(len(features_img)):
        sim_1 = cos_sim(features_img[j].cpu().numpy().flatten(), features_dist_img_1[j].cpu().numpy().flatten())
        sim_2 = cos_sim(features_img[j].cpu().numpy().flatten(), features_dist_img_2[j].cpu().numpy().flatten())

        if (sim_1 < sim_2) and (row.right_vote == 1):
            correct[j] += 1
        elif (sim_1 > sim_2) and (row.left_vote == 1):
            correct[j] += 1
        else:
            pass


print(f'Total images {total}')
for k in range(len(correct)):
    print(f"CLIP base16 NightsPerceptualDataset accuracy layer cosine {k} = {100*correct[k]/total:.3f}")