# -*- coding: utf-8 -*-
"""
I/O utilities
"""

import os
import csv


def result_save(result_path, epoch, chunk_idx, pseudo_f1, sample_f1, affiliation_f1, mAP, loss_values, classes):

    os.makedirs(os.path.dirname(result_path), exist_ok=True)    
    sample_class_names = [f'class_{i}' for i in range(classes)]
    
    header = (['epoch', 'chunck_idx', 'p_normal', 'p_ab'] # p_noraml and p_ab -> no-event and event 
            + sample_class_names
            + [f'af_{cls}' for cls in sample_class_names]
            + ["mAP"] + ['loss'])

 
    event_f1_scores = [0.0]    
    # input
    for i in range(1, len(sample_class_names)):
        key = f'class_{i}'
        if key in affiliation_f1:
            event_f1_scores.append(affiliation_f1[key]['f1'])
        else:
            event_f1_scores.append(0.0)  

    row = ([epoch] + [chunk_idx]
           + list(pseudo_f1)
           + list(sample_f1)
           + event_f1_scores
           + [mAP]
           + [loss_values])

    file_exists = os.path.exists(result_path)
    with open(result_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)
        writer.writerow(row)   