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


images = np.zeros((len(class_numers_things),13, 1,196,768))
for i, row in tqdm(class_numers_things.iterrows(), total=len(class_numers_things)):
    folder_path = images_things_path + row.ClassID
    img_path = sorted(glob.glob(folder_path + '/*'))[0]
    img = Image.open(img_path)
    img_inputs = preprocess(img)[None]
    feature_img = list(model.visual.trunk.children())[0](img_inputs)
    images[i,0,:] = feature_img.detach().numpy()
    with torch.no_grad():
        for j, block in enumerate(list(model.visual.trunk.children())[4].children()):
            feature_img = block(feature_img)
            images[i,j+1,:] = feature_img.detach().numpy()


d_12 = np.zeros((13,len(data_things),768)) 
d_13 = np.zeros((13,len(data_things),768))
d_23 = np.zeros((13,len(data_things),768))

for i, row in tqdm(data_things.iterrows(), total=len(data_things)):
    img_1 = images[row.Im1]
    img_2 = images[row.Im2]
    img_3 = images[row.Im3]

    d_12_i = np.mean((img_1 - img_2)**2, axis=(1,2)) 
    d_13_i = np.mean((img_1 - img_3)**2, axis=(1,2))
    d_23_i = np.mean((img_2 - img_3)**2, axis=(1,2))

    for j in range(13):
        d_12[j,i,:] = d_12_i[j]
        d_13[j,i,:] = d_13_i[j]
        d_23[j,i,:] = d_23_i[j]


d_12_means = d_12.mean(axis=1)  
d_13_means = d_13.mean(axis=1)
d_23_means = d_23.mean(axis=1)

d_means = np.array([d_12_means, d_13_means, d_23_means]).mean(axis=0)

d_12_norm = np.zeros((13,len(data_things),768)) 
d_13_norm = np.zeros((13,len(data_things),768))
d_23_norm = np.zeros((13,len(data_things),768))


for i in range(13):
    for j in range(768):
        d_12_norm[i,:,j] = d_12[i,:,j] / d_means[i,j]   
        d_13_norm[i,:,j] = d_13[i,:,j] / d_means[i,j]  
        d_23_norm[i,:,j] = d_23[i,:,j] / d_means[i,j]    


d_12_norm_fin = np.sqrt(d_12_norm.mean(axis=2)) 
d_13_norm_fin = np.sqrt(d_13_norm.mean(axis=2))
d_23_norm_fin = np.sqrt(d_23_norm.mean(axis=2))


total = 0
correct = np.zeros(13)

for i, row in tqdm(data_things.iterrows(), total=len(data_things)):
    total += 1

    for j in range(13):
        if (d_12_norm_fin[j,i] < d_13_norm_fin[j,i]) and (d_12_norm_fin[j,i] < d_23_norm_fin[j,i]):
            correct[j] += 1
        else:
            pass

# %%
print(f'Total images {total}')
for k in range(len(correct)):
    print(f"BiomedCLIP THINGS-odd-one-out accuracy layer {k}= {100*correct[k]/total:.3f}")