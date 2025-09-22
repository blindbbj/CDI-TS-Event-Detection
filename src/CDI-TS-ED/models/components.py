# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class GatedAttentionPool(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=64):
        super().__init__()
        self.att_gate = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),                        
            nn.Linear(hidden_dim, 1)
        )
        self.feature_proj = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU()
        )

    def forward(self, x): 
        attn_scores = self.att_gate(x)           
        attn_weights = torch.softmax(attn_scores, dim=1)  
        pooled = (x * attn_weights).sum(dim=1)   
        return self.feature_proj(pooled)         

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)].to(x.device)

class MultiDilatedTCNAttentionModulator(nn.Module):
    def __init__(self, input_dim, kernel_size=3, dilation_rates=[1, 2, 4]):
        super().__init__()
        self.attn_branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(input_dim, input_dim, kernel_size=kernel_size,
                          padding=((kernel_size - 1) * d) // 2, dilation=d),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Conv1d(input_dim, input_dim, kernel_size=1)
            )
            for d in dilation_rates
        ])
        self.sigmoid = nn.Sigmoid()

    def forward(self, x): 
        x_perm = x.permute(0, 2, 1)
        attn_scores_list = [branch(x_perm) for branch in self.attn_branches]  
        attn_scores = torch.mean(torch.stack(attn_scores_list, dim=0), dim=0) 
        attn_weights = self.sigmoid(attn_scores)
        modulated = x_perm * attn_weights 
        out = modulated.permute(0, 2, 1) 
        return out
