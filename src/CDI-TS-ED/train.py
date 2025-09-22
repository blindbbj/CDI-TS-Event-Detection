# -*- coding: utf-8 -*-

# for training
import os
import yaml
import torch
import numpy as np
from torch.optim import AdamW
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from .data import ChunkDatasetManager
from .models import Model
from .losses import Loss
from .utils.labels import decode_event_predictions
from .utils.io import result_save

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def get_ratios_from_config(config):
    try:
        if 'ratios' in config and 'variant' in config:
            ratios = config['ratios'][config['variant']]
            return ratios['chief_ratio'], ratios['label_ratio'], ratios['lead_num']
        else:
            return config['loss']['chief_ratio'], config['loss']['label_ratio'], config['data']['lead_num']
    except KeyError as e:
        raise KeyError(f"Missing key in configuration file: {e}")

class train_loop:
    def __init__(self, config_path):
        self.config = load_config(config_path)
        self.device = self._init_device()
        
        set_seed(self.config['training']['seed'])
        chief_ratio, label_ratio, lead_num = get_ratios_from_config(self.config)
        self.chunk_manager = ChunkDatasetManager(
            chunk_dir=self.config['data']['path'], 
            device=self.device, 
            batch_size=self.config['training']['batch_size'], 
            lead_num=lead_num, 
            exe=self.config['data']['name']
        )
        self.num_chunks = len(self.chunk_manager)
        self.model = Model(
            device=self.device, 
            chief_ratio=chief_ratio, 
            num_classes=self.config['model']['num_classes'], 
            num_queries=self.config['model']['num_queries'], 
            d_model=self.config['model']['d_model'], 
            nhead=self.config['model']['nhead'], 
            num_layers=self.config['model']['num_layers'], 
        ).to(self.device)
        
        self.optimizer = AdamW(
            self.model.parameters(), 
            lr=float(self.config['training']['lr']), 
            weight_decay=5e-2
        )
        
        self.loss_fn = Loss(self.device, chief_ratio, label_ratio)
        self.save_dir = self.config['save']['dir']
        os.makedirs(self.save_dir, exist_ok=True)
        
        print("Initialization complete")

    def _init_device(self):
        device = torch.device(self.config['training']['device'])
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"
        torch.set_default_device(device)
        return device

    def _load_checkpoint(self):
        checkpoint_path = os.path.join(self.save_dir, "best_model.pth")
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f"Loaded checkpoint with loss {checkpoint['loss']:.4f}")
            return checkpoint['loss']
        return float("inf")

    def _train_epoch(self, chunk_idx):
        train_loader, _ = self.chunk_manager.get_loader_by_chunk_index(chunk_idx)
        self.model.train()
        
        epoch_loss = 0.0
        
        for x, y, gy, pse, targets in tqdm(train_loader, desc=f"Train Chunk {chunk_idx+1}"):
            inputs = x.to(self.device)
            labels = {
                'gy': gy.to(self.device),
                'pse': pse.to(self.device),
                'targets': [{k: v.to(self.device) for k, v in t.items()} for t in targets]
            }
            
            self.optimizer.zero_grad()
            
            model_outputs = self.model(inputs)
            total_loss = self.loss_fn.custom(
                model_outputs['scaler_result'], 
                model_outputs['pred_boxes'], 
                model_outputs['pred_logits'], 
                labels['pse'], 
                labels['targets'], 
                model_outputs['gate_result'], 
                labels['gy'], 
                model_outputs['pgi_result']
            )
            loss = total_loss['total_loss']
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.item()
        return epoch_loss / len(train_loader)

    def train(self):
        best_loss = self._load_checkpoint()
        no_improve_count = 0
        patience = self.config['training']['patience']
        
        for epoch in range(self.config['training']['epochs']):
            print(f"\n=== Epoch {epoch + 1}/{self.config['training']['epochs']} ===")
            total_epoch_loss = 0.0
            for chunk_idx in range(self.num_chunks):
                chunk_loss = self._train_epoch(chunk_idx)
                print(f"Chunk {chunk_idx + 1} Loss: {chunk_loss:.4f}")
                total_epoch_loss += chunk_loss
            avg_epoch_loss = total_epoch_loss / self.num_chunks
            print(f"Epoch {epoch + 1} Average Loss: {avg_epoch_loss:.4f}")

            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
                no_improve_count = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': best_loss,
                }, os.path.join(self.save_dir, "best_model.pth"))
                print(f"New best model saved with loss: {best_loss:.4f}")
            else:
                no_improve_count += 1
                print(f"No improvement for {no_improve_count} epoch(s).")
            if no_improve_count >= patience:
                print(f"Early stopping triggered after {patience} epochs with no improvement.")
                break

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Train TimeET model')
    parser.add_argument('--config', type=str, required=True, help='Path to the YAML config file.')
    args = parser.parse_args()
    
    try:
        trainer = train_loop(args.config)
        trainer.train()
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()

