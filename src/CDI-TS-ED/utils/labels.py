# -*- coding: utf-8 -*-
"""
Label processing utilities
"""

import numpy as np
import torch
import torch.nn.functional as F


def generate_event_targets_np(label_array):
    B, T = label_array.shape
    targets = []

    for b in range(B):
        labels_b = label_array[b]
        events = []
        prev_label = 0
        start = None

        for t in range(T):
            curr = labels_b[t]
            if curr != 0 and curr != prev_label:
                start = t
            elif curr == 0 and prev_label != 0 and start is not None:
                end = t
                events.append((prev_label, start, end))
                start = None
            prev_label = curr

        if prev_label != 0 and start is not None:
            events.append((prev_label, start, T))

        if events:
            labels = np.array([cls for cls, _, _ in events], dtype=np.int64)
            boxes = np.array([[s / T, e / T] for _, s, e in events], dtype=np.float32)
        else:
            labels = np.zeros((0,), dtype=np.int64)
            boxes = np.zeros((0, 2), dtype=np.float32)

        targets.append({
            "labels": labels,   
            "boxes": boxes      
        })

    return targets

def decode_event_predictions(pred_boxes, pred_logits, seq_len):
    B, N, C = pred_logits.shape 
    pred_seq = torch.zeros(B, seq_len, dtype=torch.long, device=pred_logits.device)

    pred_probs = F.softmax(pred_logits, dim=-1)
    pred_labels = torch.argmax(pred_probs, dim=-1)

    for b in range(B):
        for i in range(N):
            cls = pred_labels[b, i].item()
            if cls == 0:
                continue 
            box = pred_boxes[b, i] 
            start = max(0, int(box[0].item() * seq_len))
            end = min(seq_len, int(box[1].item() * seq_len))
            pred_seq[b, start:end] = cls  
    return pred_seq
