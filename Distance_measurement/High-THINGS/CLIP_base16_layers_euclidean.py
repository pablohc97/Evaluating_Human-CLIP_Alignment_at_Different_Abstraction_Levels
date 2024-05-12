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
from transformers import AutoProcessor, CLIPVisionModel
model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch16").to(device)
processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch16")

# %%
data_things = pd.read_csv('/lustre/ific.uv.es/ml/uv075/Databases/HighLevel/THINGS/triplets/triplet_dataset/testset1_paths.csv')
class_numers_things = pd.read_csv('/lustre/ific.uv.es/ml/uv075/Databases/HighLevel/THINGS/triplets/triplet_dataset/class_name_number.csv')
things_path = '/lustre/ific.uv.es/ml/uv075/Databases/HighLevel/THINGS'
images_things_path = '/lustre/ific.uv.es/ml/uv075/Databases/HighLevel/THINGS/THINGS/Images/'

# %%
images = np.zeros((len(class_numers_things),13, 1,197,768))
for i, row in tqdm(class_numers_things.iterrows(), total=len(class_numers_things)):
    folder_path = images_things_path + row.ClassID
    img_path = sorted(glob.glob(folder_path + '/*'))[0]
    img = Image.open(img_path)
    input_img = processor(images=img, return_tensors="pt").to(device)
    features_img = model(**input_img, output_hidden_states=True).hidden_states
    for j in range(len(features_img)):
        images[i,j,:] = features_img[j].cpu().detach().numpy()
    
print(images.shape)

# %%
correct = np.zeros(13)
total = 0

for i, row in tqdm(data_things.iterrows(), total=len(data_things)):
    img_1 = images[row.Im1]
    img_2 = images[row.Im2]
    img_3 = images[row.Im3]

    d_12 = np.sqrt(np.mean((img_1 - img_2)**2, axis=(1,2,3)))
    d_13 = np.sqrt(np.mean((img_1 - img_3)**2, axis=(1,2,3)))
    d_23 = np.sqrt(np.mean((img_2 - img_3)**2, axis=(1,2,3)))

    total += 1

    for j in range(13):
        if (d_12[j] < d_13[j]) and (d_12[j] < d_23[j]):
            correct[j] += 1
        else:
            pass

# %%
print(f'Total images {total}')
for k in range(len(correct)):
    print(f"CLIP base16 THINGS-odd-one-out accuracy layer euclidean {k}= {100*correct[k]/total:.3f}")


