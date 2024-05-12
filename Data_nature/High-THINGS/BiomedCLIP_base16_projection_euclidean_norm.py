# %%
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import glob
from PIL import Image

import os
device = "cuda"


from open_clip import create_model_from_pretrained, get_tokenizer # works on open-clip-torch>=2.23.0, timm>=0.9.8

model, preprocess = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
model.eval()
tokenizer = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')


data_things = pd.read_csv('/lustre/ific.uv.es/ml/uv075/Databases/HighLevel/THINGS/triplets/triplet_dataset/testset1_paths.csv')
class_numers_things = pd.read_csv('/lustre/ific.uv.es/ml/uv075/Databases/HighLevel/THINGS/triplets/triplet_dataset/class_name_number.csv')
things_path = '/lustre/ific.uv.es/ml/uv075/Databases/HighLevel/THINGS'
images_things_path = '/lustre/ific.uv.es/ml/uv075/Databases/HighLevel/THINGS/THINGS/Images/'


images = np.zeros((len(class_numers_things), 1,196,768))
for i, row in tqdm(class_numers_things.iterrows(), total=len(class_numers_things)):
    folder_path = images_things_path + row.ClassID
    img_path = sorted(glob.glob(folder_path + '/*'))[0]
    img = Image.open(img_path)
    img_inputs = preprocess(img)[None]
    images[i,:] = model.visual(img_inputs).detach().numpy()


d_12 = np.zeros((len(data_things),768)) 
d_13 = np.zeros((len(data_things),768))
d_23 = np.zeros((len(data_things),768))

for i, row in tqdm(data_things.iterrows(), total=len(data_things)):
    img_1 = images[row.Im1]
    img_2 = images[row.Im2]
    img_3 = images[row.Im3]

    d_12_i = np.mean((img_1 - img_2)**2, axis=(0,1)) 
    d_13_i = np.mean((img_1 - img_3)**2, axis=(0,1))
    d_23_i = np.mean((img_2 - img_3)**2, axis=(0,1))


d_12_means = d_12.mean(axis=0)  
d_13_means = d_13.mean(axis=0)
d_23_means = d_23.mean(axis=0)

d_means = np.array([d_12_means, d_13_means, d_23_means]).mean(axis=0)

d_12_norm = np.zeros((len(data_things),768)) 
d_13_norm = np.zeros((len(data_things),768))
d_23_norm = np.zeros((len(data_things),768))


for j in range(768):
    d_12_norm[:,j] = d_12[:,j] / d_means[j]   
    d_13_norm[:,j] = d_13[:,j] / d_means[j]  
    d_23_norm[:,j] = d_23[:,j] / d_means[j]    


d_12_norm_fin = np.sqrt(d_12_norm.mean(axis=1)) 
d_13_norm_fin = np.sqrt(d_13_norm.mean(axis=1))
d_23_norm_fin = np.sqrt(d_23_norm.mean(axis=1))


total = 0
correct = 0

for i, row in tqdm(data_things.iterrows(), total=len(data_things)):
    total += 1

    if (d_12_norm_fin[i] < d_13_norm_fin[i]) and (d_12_norm_fin[i] < d_23_norm_fin[i]):
        correct += 1
    else:
        pass

# %%
print(f'Total images {total}')
print(f"BiomedCLIP base16 THINGS-odd-one-out accuracy projection euclidean norm = {100*correct/total:.3f}")