import cv2
from collections import defaultdict
from copy import deepcopy
import re
from typing import Callable, List, Tuple

from datasets import load_dataset
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torchvision import transforms
import torch.nn.functional as F
from tqdm.auto import tqdm
from datetime import datetime
from sklearn.utils import shuffle
import pickle
import os
import json
from torch.utils.data import DataLoader
import gc

from augments.resize import FixedHeightResize
from augments.ocrodeg import OcrodegAug

from converter import Converter
from iam_dataset import iam_collate_batch
from iam_dataset import IAMDataset
from modules import OCROnly
from CustomOCR import CustomOCR
from CustomOCR import train_ocr



import random
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

cs = json.load(open(os.path.join('ocr', 'iam_charmap.json'), 'rt'))
cs['/PAD/'] = len(cs)+1
converter = Converter(cs)



BATCH_SIZE=32
lr = 0.001
n_layers = 3
feature_dim=512
min_nb_crops = 1000
pca_sample_size = 1000
height = 64

def lambda_replacement_for_pickling(x):
    return 1 - x



augs = OcrodegAug(
    p_random_vert_pad=0.2,
    p_random_hori_pad=0.2,
    p_random_squeeze_stretch=0.2,
    p_dilation=0.2,
    p_erosion=0.2,
    p_distort_with_noise=0.2,
    p_background_noise=0.2,
)

train_trans= transforms.Compose([
        transforms.Grayscale(),
        augs,
        FixedHeightResize(height),
        transforms.ToTensor(),
        lambda_replacement_for_pickling,
])


train_ds = IAMDataset(folder="iam/train/", charset=cs, transform=train_trans, height=height)
train_ds_size = len(train_ds)


val_trans = transforms.Compose([
    transforms.Grayscale(),
    FixedHeightResize(height),
    transforms.ToTensor(),
    lambda_replacement_for_pickling,
])

valid_ds = IAMDataset(folder="iam/valid/", charset=None, transform=val_trans, height=height)
valid_dataloader = DataLoader(valid_ds, batch_size=1, shuffle=False, collate_fn=iam_collate_batch, num_workers=7)

#train_ds, _ = torch.utils.data.random_split(train_ds, [size, train_ds_size - size])

train_dataloader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=iam_collate_batch, num_workers=7)


model = OCROnly(len(cs)+1, feature_dim=feature_dim).to(device)

optimizer = torch.optim.RAdam(model.parameters(), lr=lr, weight_decay=0)
criterion = torch.nn.CTCLoss(zero_infinity=True)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=30, verbose=1)

train_ocr(model, train_dataloader, valid_dataloader, criterion, converter, [optimizer], [scheduler], "iam_results/iam_vanilla")

del model
torch.cuda.empty_cache()
gc.collect()

pca_batch = next(iter(DataLoader(train_ds, batch_size=250, shuffle=True, collate_fn=iam_collate_batch)))[0].to(device)

model = CustomOCR(len(cs)+1, feature_dim=feature_dim).to(device)
model.pca_weight_init(pca_batch, min_nb_crops, pca_sample_size)

optimizer = torch.optim.RAdam(model.parameters(), lr=lr, weight_decay=0)
criterion = torch.nn.CTCLoss(zero_infinity=True)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=30, verbose=1)

train_ocr(model, train_dataloader, valid_dataloader, criterion, converter, [optimizer], [scheduler], "iam_results/iam_pca")

del model
del pca_batch
torch.cuda.empty_cache()
gc.collect()
