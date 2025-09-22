# -*- coding: utf-8 -*-
"""
Data loading and chunk management
"""
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedShuffleSplit
from ..utils.labels import generate_event_targets_np
from .datasets import TimeSeriesFullLabelDataset

class ChunkDatasetManager:
    def __init__(self, chunk_dir, device, batch_size, lead_num, exe="none"):
        self.chunk_dir = chunk_dir
        self.device = device
        self.batch_size = batch_size
        self.lead_num = lead_num
        self.g = torch.Generator(device=self.device)
        self.exe = exe

        self.chunk_ids = sorted([
            "_".join(f.split("_")[:2]) for f in os.listdir(chunk_dir)
            if f.endswith("_label.npy")
        ])
        print(self.chunk_ids)

    def __len__(self):
        return len(self.chunk_ids)
    
    def get_loader_by_chunk_index(self, idx):
        chunk_id = self.chunk_ids[idx]
        
        data = np.load(os.path.join(self.chunk_dir, f"{chunk_id}.npy"))
        label = np.load(os.path.join(self.chunk_dir, f"{chunk_id}_label.npy")).astype(int).squeeze(-1)
        
        if self.exe == "opp":
            label[label == 4] = 3
            label[label == 5] = 4

        gaussian = np.load(os.path.join(self.chunk_dir, f"{chunk_id}_gaussian.npy"))
        print(data.shape, label.shape, gaussian.shape)

        chief = np.array([int(np.any(seq != 0)) for seq in label])

        target = generate_event_targets_np(label)

        def extract_major_label(seq_L):
            values, counts = np.unique(seq_L, return_counts=True)
            return values[np.argmax(counts)]

        y_major = [extract_major_label(seq_L) for seq_L in label]

        sss = StratifiedShuffleSplit(n_splits=1, test_size=1/9, random_state=42)
        idx = np.arange(len(y_major))
        train_idx, val_idx = next(sss.split(np.zeros(len(y_major)), y_major))

        x_tensor = torch.tensor(data, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.long)
        gaussian_tensor = torch.tensor(gaussian, dtype=torch.float32)
        chief_tensor = torch.tensor(chief, dtype=torch.long)

        train_dataset = TimeSeriesFullLabelDataset(
            x_tensor[train_idx],
            label_tensor[train_idx],
            gaussian_tensor[train_idx],
            chief_tensor[train_idx],
            [target[i] for i in train_idx]
        )

        val_dataset = TimeSeriesFullLabelDataset(
            x_tensor[val_idx],
            label_tensor[val_idx],
            gaussian_tensor[val_idx],
            chief_tensor[val_idx],
            [target[i] for i in val_idx]
        )

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=self.collate_fn, generator=self.g)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=self.collate_fn)

        return train_loader, val_loader

    def collate_fn(self, batch):
        x, y_seq, gy, domain, targets = zip(*batch)
        return (
            torch.stack(x),
            torch.stack(y_seq),
            torch.stack(gy),
            torch.stack(domain),
            list(targets)
        )
