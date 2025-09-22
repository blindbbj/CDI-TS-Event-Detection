# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from chronos import BaseChronosPipeline
import torch.nn.functional as F

from .components import GatedAttentionPool, PositionalEncoding, MultiDilatedTCNAttentionModulator
from .transformers import StandardTransformerEncoder, StandardTransformerDecoder
from torch.nn import MultiheadAttention

class StandardBlock(nn.Module):
    def __init__(self, encoder, tcn, norm):
        super().__init__()
        self.encoder = encoder
        self.tcn = tcn
        self.norm = norm

    def forward(self, x):
        x = self.encoder(x) 
        x = self.tcn(x)
        x = self.norm(x)
        return x

class Model(nn.Module): 
    def __init__(self, device, chief_ratio, num_classes=3, num_queries=10, d_model=40, nhead=8, num_layers=6):
        super().__init__()
        
        self.device = device
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.context_length = 4096
        self.dropout_rate = 0.3
        self.dropout_rate_detr = 0.2
        self.attention_dropout = 0.1

        self.chief_weights = self._initialize_class_weights(chief_ratio)
        
        self.pipeline = self._initialize_chronos_pipeline()
        self.ch_reducer = self._create_mlp_reducer()
        
        self.pos_encoder = PositionalEncoding(d_model=self.d_model)
        self.gce_pos_encoder = PositionalEncoding(d_model=self.d_model)
        self.ldi_pos_encoder = PositionalEncoding(d_model=self.d_model * 2)

        self.gce_block = self._make_standard_block(i_dim=self.d_model, detail=7)
        self.ldi_block = self._make_standard_block(i_dim=self.d_model * 2, detail=3)
        
        self.reduce_conv = nn.Conv1d(in_channels=self.d_model * 2, out_channels=self.d_model, kernel_size=1)
        self.proj = nn.Linear(self.d_model, self.d_model)
        self.gce_ldi_cross_attn = MultiheadAttention(
            embed_dim=self.d_model, num_heads=self.nhead, dropout=self.attention_dropout, batch_first=True
        )
        self.GATP = GatedAttentionPool(input_dim=self.d_model, output_dim=len(chief_ratio))
        self.pos_conv = nn.Conv1d(in_channels=self.d_model, out_channels=256, kernel_size=3, padding=1)
        
        self.conv_layer = self._create_conv_head()
        self.query_embed = nn.Embedding(self.num_queries, self.d_model)
        self.query_pos = nn.Parameter(torch.randn(1, self.num_queries, self.d_model))
        self.bbox_predictor = self._create_bbox_predictor()
        self.class_predictor = self._create_class_predictor()
        self.encoder = StandardTransformerEncoder(d_model=self.d_model, nhead=4, num_layers=3)
        self.decoder = StandardTransformerDecoder(d_model=self.d_model, nhead=4, num_layers=3)

    def _initialize_class_weights(self, chief_ratio):
        weights = torch.tensor(chief_ratio, device=self.device, dtype=torch.float32)
        weights = 1.0 / weights
        return weights / weights.sum()

    def _initialize_chronos_pipeline(self):
        pipeline = BaseChronosPipeline.from_pretrained(
            "amazon/chronos-bolt-tiny", device_map=self.device, torch_dtype=torch.float32
        )
        pipeline.model.config.chronos_config['context_length'] = self.context_length
        return pipeline

    def _create_mlp_reducer(self):
        return nn.Sequential(
            nn.LazyLinear(256), nn.ReLU(), nn.Dropout(self.dropout_rate), nn.LayerNorm(256),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(self.dropout_rate), nn.LayerNorm(128),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(self.dropout_rate), nn.LayerNorm(64),
            nn.Linear(64, self.d_model)
        )

    def _make_standard_block(self, i_dim, detail):
        standard_encoder = StandardTransformerEncoder(d_model=i_dim, nhead=self.nhead, num_layers=self.num_layers)
        tcn = MultiDilatedTCNAttentionModulator(input_dim=i_dim, kernel_size=detail, dilation_rates=[1, 2, 4, 8])
        norm = nn.LayerNorm(i_dim)
        return StandardBlock(standard_encoder, tcn, norm)

    def _create_conv_head(self):
        return nn.Sequential(
            nn.Conv1d(in_channels=self.d_model + 256, out_channels=64, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=32, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=32, out_channels=1, kernel_size=1)
        )

    def _create_bbox_predictor(self):
        return nn.Sequential(
            nn.Linear(self.d_model, 64), nn.ReLU(), nn.Dropout(self.dropout_rate_detr),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 2), nn.Sigmoid()
        )

    def _create_class_predictor(self):
        return nn.Sequential(
            nn.Linear(self.d_model, 64), nn.ReLU(), nn.Dropout(self.dropout_rate_detr),
            nn.Linear(64, self.num_classes)
        )

    def forward(self, x):
        B, L, C = x.shape
        
        with torch.no_grad():
            features = [self.pipeline.embed(x[:, :, i])[0] for i in range(C)]
            feature_cat = torch.cat(features, dim=-1).to(self.device)
        
        feature = self.ch_reducer(feature_cat)
        feature = self.pos_encoder(feature)
        feature = self.proj(feature)
        
        gce = self.gce_block(feature)
        gce = self.gce_pos_encoder(gce)
        gce = self.proj(gce)
        
        ldi_input = torch.cat([feature, gce], dim=-1)
        ldi = self.ldi_block(ldi_input)
        ldi = self.ldi_pos_encoder(ldi)
        ldi = self.reduce_conv(ldi.permute(0, 2, 1)).permute(0, 2, 1)
        ldi = self.proj(ldi)
        
        weighted_gate = self.GATP(gce)
        pred_class = torch.argmax(weighted_gate, dim=-1)
        batch_weight = self.chief_weights[pred_class]
        
        combined, _ = self.gce_ldi_cross_attn(query=ldi, key=gce, value=gce)
        raw_pos_combined = self.pos_conv(combined.transpose(1, 2)).transpose(1, 2)
        combined = torch.cat([combined, raw_pos_combined], dim=-1)
        
        gate_raw = self.conv_layer(combined.permute(0, 2, 1)).permute(0, 2, 1)
        gate = torch.sigmoid(gate_raw.squeeze(-1)) 
        
        adjusted_gate = gate * batch_weight.unsqueeze(1)
        gate_exp = adjusted_gate.unsqueeze(-1)
        final_out = gate_exp * ldi + (1 - gate_exp) * gce
        
        query_embed = self.query_embed.weight.unsqueeze(0).repeat(B, 1, 1) + self.query_pos
        pos_out = self.decoder(tgt=query_embed, memory=final_out)
        
        pred_boxes = self.bbox_predictor(pos_out)
        pred_logits = self.class_predictor(pos_out)
        
        return {
            'scaler_result': weighted_gate,
            'gate_result': adjusted_gate,
            'pred_boxes': pred_boxes,
            'pred_logits': pred_logits,
            'ldi': ldi,
            'gce': gce,
            'pgi_result': raw_pos_combined,
            'final_out': final_out
        }