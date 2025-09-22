# -*- coding: utf-8 -*-
import torch
from scipy.optimize import linear_sum_assignment

class DetectionMatcher:
    def __init__(self, cost_class=1.0, cost_center=5.0, cost_length=1.0):
        self.cost_class = cost_class
        self.cost_center = cost_center
        self.cost_length = cost_length
        self.cost_iou   = 2.0

    def iou_1d(self, boxes1, boxes2):
        N, M = boxes1.size(0), boxes2.size(0)
        left = torch.max(boxes1[:, None, 0], boxes2[None, :, 0])
        right = torch.min(boxes1[:, None, 1], boxes2[None, :, 1])
        inter = (right - left).clamp(min=0)

        len1 = (boxes1[:, 1] - boxes1[:, 0]).clamp(min=1e-6)
        len2 = (boxes2[:, 1] - boxes2[:, 0]).clamp(min=1e-6)
        union = len1[:, None] + len2[None, :] - inter
        return inter / union

    @torch.no_grad()
    def __call__(self, pred_logits, pred_boxes, targets):
        B, N, C = pred_logits.shape
        indices = []

        for b in range(B):
            prob = pred_logits[b].softmax(-1)
            boxes = pred_boxes[b]
            tgt_labels = targets[b]["labels"]
            tgt_boxes = targets[b]["boxes"]
            M = tgt_labels.shape[0]

            if M == 0:
                indices.append((torch.empty(0, dtype=torch.long),
                                torch.empty(0, dtype=torch.long)))
                continue

            cost_class = torch.stack([-prob[:, lbl] for lbl in tgt_labels], dim=1)

            pred_center = (boxes[:, 0] + boxes[:, 1]) / 2
            tgt_center = (tgt_boxes[:, 0] + tgt_boxes[:, 1]) / 2
            cost_center = torch.abs(pred_center[:, None] - tgt_center[None, :])

            pred_len = (boxes[:, 1] - boxes[:, 0]).abs()
            tgt_len = (tgt_boxes[:, 1] - tgt_boxes[:, 0]).abs()
            cost_length = torch.abs(pred_len[:, None] - tgt_len[None, :])

            total_cost = (
                self.cost_class * cost_class +
                self.cost_center * cost_center +
                self.cost_length * cost_length
            ).cpu()

            src_oto, tgt_oto = linear_sum_assignment(total_cost)
            src_oto = torch.as_tensor(src_oto, dtype=torch.long)
            tgt_oto = torch.as_tensor(tgt_oto, dtype=torch.long)

            indices.append((src_oto, tgt_oto))

        return indices
