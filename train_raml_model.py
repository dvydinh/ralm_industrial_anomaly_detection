import os
import sys
import json
import time
import random
import argparse
import shutil
import zipfile
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
from glob import glob
from sklearn.metrics import roc_auc_score
from collections import defaultdict
from typing import List, Tuple, Dict, Optional

# ========================================================================================
# [CORE CLASSES - AUTHENTIC IMPLEMENTATION]
# ========================================================================================

# --- COMPONENT 2: ATTENTION & PYRAMID ---
class CrossScaleAttention(nn.Module):
    def __init__(self, dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.num_heads, self.head_dim = num_heads, dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.q_proj, self.k_proj, self.v_proj = nn.Linear(dim, dim), nn.Linear(dim, dim), nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout, self.norm = nn.Dropout(dropout), nn.LayerNorm(dim)
        
    def forward(self, query, key_value):
        B, D = query.shape
        _, N, _ = key_value.shape
        q = self.q_proj(query).unsqueeze(1).view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key_value).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(key_value).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        out = (self.dropout(attn) @ v).transpose(1, 2).reshape(B, D)
        return self.norm(query + self.out_proj(out))

class MultiScalePyramid(nn.Module):
    def __init__(self, feature_dim=512, hidden_dim=256):
        super().__init__()
        self.s1_proj, self.s2_proj, self.s3_proj = nn.Linear(feature_dim, hidden_dim), nn.Linear(feature_dim, hidden_dim), nn.Linear(feature_dim, hidden_dim)
        self.attn12, self.attn23, self.attn31 = CrossScaleAttention(hidden_dim), CrossScaleAttention(hidden_dim), CrossScaleAttention(hidden_dim)
        self.fusion = nn.Sequential(nn.Linear(hidden_dim*3, hidden_dim*2), nn.GELU(), nn.Dropout(0.1), nn.Linear(hidden_dim*2, hidden_dim))
        self.norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, g, feat_s2, feat_s3):
        f1, f2, f3 = self.s1_proj(g), self.s2_proj(feat_s2.mean(1)), self.s3_proj(feat_s3.mean(1))
        f1a = self.attn12(f1, self.s2_proj(feat_s2))
        f2a = self.attn23(f2, self.s3_proj(feat_s3))
        f3a = self.attn31(f3, f1.unsqueeze(1))
        return F.normalize(self.norm(self.fusion(torch.cat([f1a, f2a, f3a], dim=-1))), p=2.0, dim=1)

# --- COMPONENT 3: TRACKER & LOSS ---
class AutoDifficultyTracker:
    def __init__(self, momentum=0.7, weight_scale=0.5):
        self.momentum, self.weight_scale = momentum, weight_scale
        self.running_loss, self.count = {}, {}
    def update(self, cat, loss):
        if cat not in self.running_loss: self.running_loss[cat] = loss
        else: self.running_loss[cat] = self.momentum * self.running_loss[cat] + (1 - self.momentum) * loss
    def get_weight(self, cat):
        if not self.running_loss or len(self.running_loss) < 2: return 1.0
        if cat not in self.running_loss: return 1.0 # Safety for first batch of new category
        g_loss = np.mean(list(self.running_loss.values()))
        return 1.0 if g_loss < 1e-6 else float(np.clip(1.0 + self.weight_scale * (self.running_loss[cat] - g_loss) / g_loss, 0.5, 2.0))

class MACCLLoss(nn.Module):
    def __init__(self, feature_dim=256, margin_base=0.5, lambda_sigma=0.3, lambda_resolution=0.3, 
                 original_resolution=900, model_resolution=224, temperature=0.07, 
                 alpha=1.0, beta=1.0, gamma=0.5):
        super().__init__()
        self.register_buffer('normal_center', torch.zeros(feature_dim))
        self.register_buffer('running_sigma', torch.tensor(1.0))
        self.margin_base = margin_base
        self.lambda_sigma = lambda_sigma
        self.lambda_resolution = lambda_resolution
        self.resolution_ratio = model_resolution / original_resolution
        self.temperature = temperature
        self.alpha, self.beta, self.gamma = alpha, beta, gamma
        self.difficulty_tracker = AutoDifficultyTracker()
        
    def compute_center_loss(self, features, normal_mask):
        if normal_mask.sum() == 0: return torch.tensor(0.0, device=features.device), torch.zeros(features.size(0), device=features.device)
        dist_sq = ((features - self.normal_center) ** 2).sum(dim=1)
        loss_raw = dist_sq * normal_mask.float()
        if normal_mask.sum() > 0:
            with torch.no_grad(): self.normal_center.copy_(0.8 * self.normal_center + 0.2 * features[normal_mask].mean(dim=0))
        return loss_raw.sum() / (normal_mask.sum() + 1e-8), loss_raw

    def compute_margin_loss(self, features, anomaly_mask, normal_mask):
        with torch.no_grad():
            if normal_mask.sum() > 0: self.running_sigma.copy_(0.9 * self.running_sigma + 0.1 * features[normal_mask].std())
        m_adaptive = self.margin_base + self.lambda_sigma * self.running_sigma + self.lambda_resolution * (1 - self.resolution_ratio)
        
        dist_to_center = torch.norm(features - self.normal_center, dim=1)
        
        loss_raw = F.relu(m_adaptive - dist_to_center) * anomaly_mask.float()
        return (loss_raw.sum() / anomaly_mask.sum()) if anomaly_mask.sum() > 0 else torch.tensor(0.0, device=features.device), loss_raw, m_adaptive.item()

    def compute_contrastive_loss(self, features, labels):
        if labels.sum() == 0 or (1 - labels).sum() == 0: return torch.tensor(0.0, device=features.device), torch.zeros(features.size(0), device=features.device)
        features = F.normalize(features, p=2.0, dim=1) # Explicit p=2.0
        sim_matrix = torch.mm(features, features.t()) / self.temperature
        mask_pos = (labels.view(-1, 1) == labels.view(1, -1)).float(); mask_pos.fill_diagonal_(0)
        mask_neg = (labels.view(-1, 1) != labels.view(1, -1)).float()
        raw_losses = torch.zeros(features.size(0), device=features.device)
        for i in range(features.size(0)):
            pos_sim, neg_sim = sim_matrix[i, mask_pos[i].bool()], sim_matrix[i, mask_neg[i].bool()]
            if len(pos_sim) > 0 and len(neg_sim) > 0:
                raw_losses[i] = -torch.log(torch.exp(pos_sim).sum() / (torch.exp(pos_sim).sum() + torch.exp(neg_sim).sum() + 1e-8))
        return (raw_losses.sum() / (raw_losses > 0).sum()) if (raw_losses > 0).sum() > 0 else torch.tensor(0.0, device=features.device, requires_grad=True), raw_losses

    def forward(self, features, labels, categories=None):
        normal_mask, anomaly_mask = (labels == 0), (labels == 1)
        l_center, r_center = self.compute_center_loss(features, normal_mask)
        l_margin, r_margin, adaptive_m = self.compute_margin_loss(features, anomaly_mask, normal_mask)
        l_con, r_con = self.compute_contrastive_loss(features, labels)
        
        raw_total = self.alpha * r_center + self.beta * r_margin + self.gamma * r_con
        if categories:
            weights = torch.tensor([self.difficulty_tracker.get_weight(c) for c in categories], device=features.device)
            for c, l in zip(categories, raw_total.detach().cpu().numpy()):
                self.difficulty_tracker.update(c, float(l))
            total_loss = (raw_total * weights).mean()
        else: total_loss = raw_total.mean()
        
        return total_loss, {'total': total_loss, 'adaptive_margin': adaptive_m, 'running_sigma': self.running_sigma.item()}

