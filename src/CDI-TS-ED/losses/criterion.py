# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F
from chronos import BaseChronosPipeline
from .matcher import DetectionMatcher

class Loss:
    def __init__(self, device, pseudo_ratio, label_ratio, 
                 matcher_cost_cls=1.0, matcher_cost_bbox=5.0):

        self.device = device
        self.pseudo_ratio = pseudo_ratio
        self.label_ratio = label_ratio
        self.matcher = DetectionMatcher(matcher_cost_cls, matcher_cost_bbox)

        self.pipeline = BaseChronosPipeline.from_pretrained(
            "amazon/chronos-bolt-tiny",
            device_map='cuda:0', 
            torch_dtype=torch.float32
        )
        self.pipeline.model.config.chronos_config['context_length'] = 4096

        self.label_weights = torch.tensor(self.label_ratio, device=self.device, dtype=torch.float32)
        self.label_weights = 1.0 / self.label_weights
        self.label_weights /= self.label_weights.sum()
        
        self.pseudo_weights = torch.tensor(self.pseudo_ratio, device=self.device, dtype=torch.float32)
        self.pseudo_weights = 1.0 / self.pseudo_weights
        self.pseudo_weights /= self.pseudo_weights.sum()

    def custom(self, pred_gate, pred_boxes, pred_logits, pse, targets, adjusted_gate, gy, raw_pos_combined):
        B, N, C = pred_logits.shape

        def cosine_alignment_loss(x, y):
            x_norm = F.normalize(x, dim=-1)
            y_norm = F.normalize(y, dim=-1)
            cosine_sim = (x_norm * y_norm).sum(dim=-1)
            return 1 - cosine_sim.mean()

        with torch.no_grad():
            gy_list = []
            feature, _ = self.pipeline.embed(gy.squeeze(-1)) 
            gy_list.append(feature)

        gy_list = torch.cat(gy_list, dim=-1).to(self.device)
        cos_loss = cosine_alignment_loss(raw_pos_combined, gy_list)
        bce = F.cross_entropy(pred_gate, pse, weight=self.pseudo_weights)
        indices = self.matcher(pred_logits, pred_boxes, targets)

        pred_logits_flat = pred_logits.view(-1, C)
        pred_boxes_flat = pred_boxes.view(-1, 2)
        tgt_labels_full = torch.zeros(B * N, dtype=torch.long, device=self.device)

        src_all, tgt_labels_all, tgt_boxes_all = [], [], []
        for b, (src_oto, tgt_oto) in enumerate(indices):
            if len(src_oto) > 0:
                tgt_labels = targets[b]["labels"][tgt_oto].to(self.device)
                tgt_labels_full[b * N + src_oto] = tgt_labels
                src_all.append(src_oto + b * N)
                tgt_labels_all.append(tgt_labels)
                tgt_boxes_all.append(targets[b]["boxes"][tgt_oto])

        loss_det_cls = F.cross_entropy(pred_logits_flat, tgt_labels_full, weight=self.label_weights)
        if len(src_all) > 0:
            src_all = torch.cat(src_all)
            tgt_labels_all = torch.cat(tgt_labels_all).to(self.device)
            tgt_boxes_all = torch.cat(tgt_boxes_all).to(self.device)
            pred_box_match = pred_boxes_flat[src_all]
            is_event = tgt_labels_all != 0
            if is_event.any():
                loss_det_bbox = F.l1_loss(pred_box_match[is_event], tgt_boxes_all[is_event])
            else:
                loss_det_bbox = torch.tensor(0., device=self.device)
        else:
            loss_det_bbox = torch.tensor(0., device=self.device)

        total_loss = 0.1 * bce + 0.7 * (2.0 * loss_det_cls + 5.0 * loss_det_bbox) + 0.2 * cos_loss
        loss = {"total_loss": total_loss, "bce": bce, "cos_loss": cos_loss}
        return loss
