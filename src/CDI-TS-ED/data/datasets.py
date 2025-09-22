# -*- coding: utf-8 -*-
"""
Dataset classes
"""
import torch
from torch.utils.data import Dataset

class TimeSeriesFullLabelDataset(Dataset):
    def __init__(self, x, y_seq, gaussian_label, domain_labels, detection_targets):
        self.x = x
        self.y_seq = y_seq
        self.gaussian_label = gaussian_label
        self.domain_labels = domain_labels
        self.detection_targets = detection_targets

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return (
            self.x[idx],                      
            self.y_seq[idx],                  
            self.gaussian_label[idx],
            self.domain_labels[idx],
            {
                "labels": torch.tensor(self.detection_targets[idx]["labels"], dtype=torch.long),
                "boxes": torch.tensor(self.detection_targets[idx]["boxes"], dtype=torch.float32)
            }
        )
