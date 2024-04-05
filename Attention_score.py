import os
import torch
import argparse

import numpy as np
import json
import h5py
import sys
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
import openslide
import matplotlib as mpl
from matplotlib import patches
sys.path.append('/Data/yangml/Important_script/WSIimmune')
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import auc as calc_auc
from sklearn.preprocessing import label_binarize
from model.model_ImmuSlide import ImmuSlide
import math

from utils.utils import get_optim, print_network,calculate_error, get_loader, get_simple_loader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_dict = {"dropout": False, 'n_classes': 2}
model = ImmuSlide(**model_dict)
model.eval()
model.load_state_dict(torch.load('/Data/yangml/Project/TCGA/Public7_TCGA_Part2/lung/data/LUSC/late_stage_lusc_slide/SVS/Trainning/results/s_1_checkpoint.pt'))
sample="/Data/yangml/Project/LUSC/X101SC24012865-Z01-J001/Result_X101SC24012865-Z01-J001/Slide/Test/EB22-04193-A1.h5.Feature.h5"

with h5py.File(sample, 'r') as hdf5_file:
    features = hdf5_file['Features'][:]
features = torch.from_numpy(features)
with torch.no_grad():
    Result = model(features,0)
    AttentionScore=model(features,0,attention_only=True)

samplecoord = "/Data/yangml/Project/LUSC/X101SC24012865-Z01-J001/Result_X101SC24012865-Z01-J001/Slide/patch/EB22-04193-A1.h5"
with h5py.File(samplecoord, 'r') as hdf5_file:
    coords = hdf5_file['coords'][:]

slide="/Data/yangml/Project/LUSC/X101SC24012865-Z01-J001/Result_X101SC24012865-Z01-J001/Slide/EB22-04193-A1.svs"

slide=openslide.OpenSlide(slide)
colorDict = {'Normal':sns.light_palette("seagreen", as_cmap=True)}
             # 'Tumor':sns.light_palette("palegreen", as_cmap=True)}
def Get_heatmap(attentionScore,coords,colorDict,slide,level,heatmap_name,patchSize=224):

    print(attentionScore)

    Whole_slide_RGB = slide.read_region((0, 0), level, slide.level_dimensions[level]).convert('RGB')

    Whole_slide_RGB = np.array(Whole_slide_RGB)
    Whole_slide_dim = slide.level_dimensions[level]
    print(Whole_slide_dim)

    sample_factor = slide.level_downsamples[level]

    Whole_slide_arr = np.zeros(Whole_slide_dim)

    fig, ax = plt.subplots()

    ax.set_xlabel('Width')
    ax.set_ylabel('Height')

    ax.imshow(Whole_slide_RGB,alpha= 0.5)
    for i in range(coords.shape[0]):
        coord = coords[i]
        tileX = int(np.ceil(coord[0] / sample_factor))
        tileY = int(np.ceil(coord[1] / sample_factor))
        tile_H = int(np.ceil(patchSize/sample_factor))
        tile_W = int(np.ceil(patchSize/sample_factor))
        probs = attentionScore[i]
        # max_prob=max(probs)
        camp = colorDict['Normal']
        print(tileX,tileY)
        print(tile_H,tile_W)
        # Whole_slide_arr[tileX:tileX+tile_W,tileY:tileY+tile_H]=probs
        cmo = plt.cm.get_cmap('viridis')
        # print(names[probs.index(max(probs))])
        # print(max(probs))
        # plt.gca().add_patch(plt.Rectangle((tileX, tileY), tile_W, tile_H, color=mpl.colors.to_hex(camp(max_prob))))
        ax.add_patch(patches.Rectangle((tileX, tileY), tile_W, tile_H, color=mpl.colors.to_hex(camp(probs)),alpha=0.2))
    # ax.imshow(Whole_slide_arr,cmap=cmo)
    tem=0.9
    for i in colorDict.keys():
        tem = tem - 0.1
        ax1 = fig.add_axes([0.95, tem, 0.16, 0.035])
        norm = mpl.colors.Normalize(vmin=0, vmax=1)
        sm1 = mpl.cm.ScalarMappable(norm=norm, cmap=colorDict[i])
        sm1.set_array([])
        cb1 = plt.colorbar(sm1, cax=ax1, orientation='horizontal', label=i, ticklocation='top')
        cb1.set_ticks([])

    plt.savefig(heatmap_name, bbox_inches='tight')
    plt.show()

AttentionScore = torch.softmax(AttentionScore,dim=0)
AttentionScore = AttentionScore.detach().numpy()

def sigmoid(x):
    return 1.0/(1+np.exp(-x))

def standardization(data):
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    return (data - mu) / sigma
AttentionScore1= standardization(AttentionScore)
SigmodAtten = sigmoid(AttentionScore1)
np.percentile(SigmodAtten,(25,50,90),interpolation='midpoint')

coordsHighest = coords[np.where(SigmodAtten>0.5)[0]]
for i in range(coordsHighest.shape[0]):
    coord = coordsHighest[i]
    patchimg = slide.read_region(coord, 0, (224, 224)).convert('RGB')
    patchimg.save(('./HighAttention1/' + str(coord[0]) + '_' + str(coord[1]) + '.jpg'))

# SigmodAtten[np.where(SigmodAtten >=0.49815965)] = 0.95
# SigmodAtten[np.where(SigmodAtten <0.49815965)] = 0.01

SigmodAttenSelect = SigmodAtten[np.where(SigmodAtten <=0.49186534)[0]]
coordsSelect=coords[np.where(SigmodAtten <=0.49186534)[0]]

coord = (28923,28677)
patchimg1=slide.read_region(coord,0,(1024,1024)).convert('RGB')
patchimg1.save(('./HighAttention1/' + str(coord[0]) + '_' + str(coord[1]) + '.jpg'))

Get_heatmap(SigmodAttenSelect,coordsSelect,colorDict,slide,level=2,heatmap_name="Nonresponder")