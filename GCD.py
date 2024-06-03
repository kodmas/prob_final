import os
import sys
import json
import numpy as np
import random
import nltk
import math
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from mingpt.utils import set_seed, setup_logging, CfgNode as CN

class GCDDataset(Dataset):
    @staticmethod
    def get_default_config():
        C = CN()
        C.ndigit = 2
        return C
    
    def is_prime(self, x):
        if x > 1:
            for i in range(2, x):
                if (x % i) == 0:
                    return False
        else:
            return False
        return True
    
    def gcd(self):
        data = []
        for a in range(1, 100):
            for b in range(a + 1, 100):
                c = math.gcd(a, b)
                res = f"{a:02}{b:02}{c:02}"
                data.append([int(x) for x in res])
        return data
    
    def __init__(self, config, split, seed):
        self.seed = seed
        self.config = config
        self.split = split  # train/test

        data = self.gcd()

        # Shuffle the entire data with fixed seed for consistent splitting
        random.Random(seed).shuffle(data)

        # Split data into train/test datasets
        num_test = min(int(len(data) * 0.2), 500)  # 20% of the whole dataset, or only up to 500
        test_data = data[:num_test]
        train_data = data[num_test:]

        # Apply the pattern after the split if required
        if split == 'train':
            # sort the data by abs(a) in descending order
            train_data = sorted(train_data, key=lambda x: abs(x[0]), reverse=True)
            self.data = train_data
        elif split == 'test':
            self.data = test_data
        
        # Convert the data to tensor
        self.ixes = torch.tensor(self.data, dtype=torch.long)
        

    def get_vocab_size(self):
        return 10  # digits 0..9

    def get_block_size(self):
        return 2 * self.config.ndigit + 2 * (self.config.ndigit - 1) - 1

    def __len__(self):
        return self.ixes.size(0)

    def __getitem__(self, idx):
        ndigit = self.config.ndigit
        x = self.ixes[idx][:-1]
        y = self.ixes[idx][1:].clone()  # predict the next token in the sequence
        y[:ndigit * 2 - 1] = -1  # we will only train in the output locations. -1 will mask loss to zero
        return x, y

def eval_split(device, model, dataset):
    ndigit = dataset.config.ndigit
    loader = DataLoader(dataset, batch_size=32, num_workers=0, drop_last=False)
    total_correct = 0
    for _, (x, y) in enumerate(loader):
        x = x.to(device)
        y = y.to(device)
        d1d2 = x[:, :ndigit * 2]
        d3_gt = y[:, ndigit * 2 - 1:]
        d1d2d3 = model.generate(d1d2, ndigit, do_sample=False)  # using greedy argmax, not sampling
        d3_pred = d1d2d3[:, ndigit * 2:]
        correct = torch.sum(torch.all(d3_pred == d3_gt, dim=1))
        total_correct += correct
    return total_correct / len(dataset)
