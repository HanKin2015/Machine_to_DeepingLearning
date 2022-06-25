import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from VGG import vgg16
from MalwareDataset import MalwareDataset, MalwareTrainDataset, MalwareTestDataset
from Configure import Configure
import sys
import pandas as pd

df = pd.DataFrame()
dict_res = {"Id": 'app', "Prediction1": 0, "Prediction2": 1}
df = df.append(dict_res, ignore_index=True)
df = df.append(dict_res, ignore_index=True)
df = df.append(dict_res, ignore_index=True)
df.to_csv('result.csv', index=0)