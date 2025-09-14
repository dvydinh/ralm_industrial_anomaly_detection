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

