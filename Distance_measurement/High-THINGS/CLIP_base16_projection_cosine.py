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
from transformers import AutoProcessor, CLIPVisionModelWithProjection
model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch16").to(device)
processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch16")

# %%
data_things = pd.read_csv('/lustre/ific.uv.es/ml/uv075/Databases/HighLevel/THINGS/triplets/triplet_dataset/testset1_paths.csv')
class_numers_things = pd.read_csv('/lustre/ific.uv.es/ml/uv075/Databases/HighLevel/THINGS/triplets/triplet_dataset/class_name_number.csv')
things_path = '/lustre/ific.uv.es/ml/uv075/Databases/HighLevel/THINGS'
images_things_path = '/lustre/ific.uv.es/ml/uv075/Databases/HighLevel/THINGS/THINGS/Images/'

# %%
images = np.zeros((len(class_numers_things),1,512))
for i, row in tqdm(class_numers_things.iterrows(), total=len(class_numers_things)):
    folder_path = images_things_path + row.ClassID
    img_path = sorted(glob.glob(folder_path + '/*'))[0]
    img = Image.open(img_path)
    input_img = processor(images=img, return_tensors="pt").to(device)
    features_img = model(**input_img).image_embeds
    images[i,:] = features_img.cpu().detach().numpy()
    
print(images.shape)

# %%
correct = 0
total = 0

from numpy import dot
from numpy.linalg import norm

def cos_sim(a,b):
    return dot(a, b)/(norm(a)*norm(b))

for i, row in tqdm(data_things.iterrows(), total=len(data_things)):
    img_1 = images[row.Im1]
    img_2 = images[row.Im2]
    img_3 = images[row.Im3]

    sim_12 = cos_sim(img_1.flatten(), img_2.flatten())
    sim_13 = cos_sim(img_1.flatten(), img_3.flatten())
    sim_23 = cos_sim(img_2.flatten(), img_3.flatten())

    total += 1

    if (sim_12 > sim_13) and (sim_12 > sim_23):
        correct += 1
    else:
        pass

# %%
print(f'Total images {total}')
print(f"CLIP base16 THINGS-odd-one-out accuracy projection cosine = {100*correct/total:.3f}")


