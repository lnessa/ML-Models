import os

import torch 
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import Dataset 
from math import floor
from functools import reduce


heatmap_dict = {
    "human": ("./datasets/human/", "./datasets/human_annotation/"),
    "glasses": ("./datasets/glasses/", "./datasets/glasses_annotation/")
}
    
class HeatMaps(Dataset):
    def __init__(self, mode = "test", folder_key = "glasses"):
        data_dir, anno_dir = heatmap_dict[folder_key]
        data_filenames = os.listdir(data_dir)

        datums = [ pd.read_csv(data_dir + fname) for fname in data_filenames ]
        annots = [ pd.read_csv(anno_dir + fname) for fname in data_filenames ]

        data_reducer = lambda acc, a: torch.cat((acc, torch.from_numpy(a.iloc[:, :(21 * 32)].to_numpy())), 0)
        label_reducer = lambda acc, a: torch.cat((acc, torch.from_numpy(a.iloc[:, 1:].to_numpy())), 0)

        data = reduce(data_reducer, datums, torch.empty(0)).to(torch.float32)
        labels = reduce(label_reducer, annots, torch.empty(0)).to(torch.int64)

        n_test = floor(data.size(0) * 0.2)

        self.data = data[:n_test, :] if mode == 'test' else data[n_test:, :]
        self.labels = labels[:n_test, :] if mode == 'test' else labels[n_test:, :]

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, idx):
        data = self.data[idx, :].reshape(1, 21, 32)
        label = F.one_hot(self.labels[idx, :].reshape(21, 32), 4).permute(2, 0, 1).to(torch.float32)
        return (data, label)
    


annotation_dict = {
    "1_coldglass_5mins.csv":    (4, 4, 5, 5, 2),
    "1_hotglass_5mins.csv":     (14, 16, 16, 18, 3),
    "2_coldglass_5mins.csv":    (14, 11, 16, 13, 2),
    "2_hotglass_5min.csv":      (8, 13, 10, 14, 3),
    "3_coldglass_5mins.csv":    (21, 5, 23, 6, 2),
    "3_hotglass_5min.csv":      (24, 2, 26, 5, 3),
    "4_coldglass_5mins.csv":    (27, 16, 30, 18, 2),
    "4_hotglass_10mins.csv":    (3, 10, 4, 12, 3),
    "5_coldglass_5mins.csv":    (4, 7, 5, 8, 2),
    "5_hotglass_5mins.csv":     (23, 6, 25, 7, 3)
}

