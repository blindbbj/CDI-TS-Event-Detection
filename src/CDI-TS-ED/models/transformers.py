# -*- coding: utf-8 -*-
import torch.nn as nn

class StandardTransformerEncoder(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dropout=0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=d_model * 4, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

    def forward(self, src):
        return self.transformer_encoder(src)

class StandardTransformerDecoder(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dropout=0.1):
        super().__init__()
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward=d_model * 4, dropout=dropout, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers)

    def forward(self, tgt, memory):
        return self.transformer_decoder(tgt, memory)
