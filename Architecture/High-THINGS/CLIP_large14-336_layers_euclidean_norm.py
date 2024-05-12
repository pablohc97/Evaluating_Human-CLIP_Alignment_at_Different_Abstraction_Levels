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
model = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14-336").to(device)
processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14-336")

# %%
data_things = pd.read_csv('/lustre/ific.uv.es/ml/uv075/Databases/HighLevel/THINGS/triplets/triplet_dataset/testset1_paths.csv')
class_numers_things = pd.read_csv('/lustre/ific.uv.es/ml/uv075/Databases/HighLevel/THINGS/triplets/triplet_dataset/class_name_number.csv')
things_path = '/lustre/ific.uv.es/ml/uv075/Databases/HighLevel/THINGS'
images_things_path = '/lustre/ific.uv.es/ml/uv075/Databases/HighLevel/THINGS/THINGS/Images/'

# %%
images = np.zeros((len(class_numers_things),25, 1,577,1024))
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
d_12 = np.zeros((25,len(data_things),1024)) 
d_13 = np.zeros((25,len(data_things),1024))
d_23 = np.zeros((25,len(data_things),1024))

for i, row in tqdm(data_things.iterrows(), total=len(data_things)):
    img_1 = images[row.Im1] # dim = 25,1,577,1024
    img_2 = images[row.Im2]
    img_3 = images[row.Im3]

    d_12_i = np.mean((img_1 - img_2)**2, axis=(1,2)) # dim = 25,1024
    d_13_i = np.mean((img_1 - img_3)**2, axis=(1,2))
    d_23_i = np.mean((img_2 - img_3)**2, axis=(1,2))

    for j in range(25):
        d_12[j,i,:] = d_12_i[j]
        d_13[j,i,:] = d_13_i[j]
        d_23[j,i,:] = d_23_i[j]


d_12_means = d_12.mean(axis=1)  # dim = 25,1024
d_13_means = d_13.mean(axis=1)
d_23_means = d_23.mean(axis=1)

d_means = np.array([d_12_means, d_13_means, d_23_means]).mean(axis=0)

d_12_norm = np.zeros((25,len(data_things),1024)) 
d_13_norm = np.zeros((25,len(data_things),1024))
d_23_norm = np.zeros((25,len(data_things),1024))

for i in range(25):
    for j in range(1024):
        d_12_norm[i,:,j] = d_12[i,:,j] / d_means[i,j]  
        d_13_norm[i,:,j] = d_13[i,:,j] / d_means[i,j]  
        d_23_norm[i,:,j] = d_23[i,:,j] / d_means[i,j]  


d_12_norm_fin = np.sqrt(d_12_norm.mean(axis=2))  # dim = 25,num_img
d_13_norm_fin = np.sqrt(d_13_norm.mean(axis=2))
d_23_norm_fin = np.sqrt(d_23_norm.mean(axis=2))



total = 0
correct = np.zeros(25)

for i, row in tqdm(data_things.iterrows(), total=len(data_things)):
    total += 1

    for j in range(25):
        if (d_12_norm_fin[j,i] < d_13_norm_fin[j,i]) and (d_12_norm_fin[j,i] < d_23_norm_fin[j,i]):
            correct[j] += 1
        else:
            pass


# %%
print(f'Total images {total}')
for k in range(len(correct)):
    print(f"CLIP large14-336 THINGS-odd-one-out accuracy layer {k}= {100*correct[k]/total:.3f}")


