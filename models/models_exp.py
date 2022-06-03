# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 21:26:09 2022

@author: AustinHsu
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super().__init__()
        hidden_dim = in_channels*expand_ratio
        if expand_ratio != 1:
            self.exp_1x1 = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=hidden_dim,
                    kernel_size=1,
                    ),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                )
        else:
            self.exp_1x1 = nn.Identity()
        self.conv_3x3 = nn.Sequential(
            nn.Conv2d(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                stride=stride,
                kernel_size=3,
                groups=hidden_dim,
                padding=(1,1),
                padding_mode='zeros',
                ),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(),
            )
        self.red_1x1 = nn.Sequential(
            nn.Conv2d(
                in_channels=hidden_dim,
                out_channels=out_channels,
                kernel_size=1,
                ),
            nn.BatchNorm2d(out_channels),
            )
        self.use_res_connect = stride == 1 and in_channels == out_channels
        
    def forward(self, x):
        out = self.exp_1x1(x)
        out = self.conv_3x3(out)
        out = self.red_1x1(out)
        if self.use_res_connect:
            return x + out
        else:
            return out

class LinearSelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.qkv_proj = nn.Conv2d(
            in_channels=embed_dim,
            out_channels=1 + (2*embed_dim),
            kernel_size=1,
            stride=1,
            )
        self.out_proj = nn.Conv2d(
            in_channels=embed_dim,
            out_channels=embed_dim,
            kernel_size=1,
            stride=1,
            )
        
    def forward(self, x):
        qkv = self.qkv_proj(x)
        query, key, value = torch.split(
            qkv, split_size_or_sections=[1, self.embed_dim, self.embed_dim], dim=1
            )
        
        context_scores = F.softmax(query, dim=-1)
        context_vector = key* context_scores
        context_vector = torch.sum(context_vector, dim=-1, keepdim=True)
        
        out = F.relu(value) * context_vector.expand_as(value)
        out = self.out_proj(out)
        return out

class LinearAttnFFN(nn.Module):
    def __init__(self, embed_dim, ffn_latent_dim):
        super().__init__()
        self.pre_norm_attn = nn.Sequential(
            nn.GroupNorm(
                num_groups=1, 
                num_channels=embed_dim, 
                eps=1e-5, 
                affine=True,
                ),
            LinearSelfAttention(embed_dim=embed_dim),
            )
        self.pre_norm_ffn = nn.Sequential(
            nn.GroupNorm(
                num_groups=1, 
                num_channels=embed_dim, 
                eps=1e-5, 
                affine=True,
                ),
            nn.Conv2d(
                in_channels=embed_dim,
                out_channels=ffn_latent_dim,
                kernel_size=1,
                stride=1,
                ),
            nn.SiLU(),
            nn.Conv2d(
                in_channels=ffn_latent_dim,
                out_channels=embed_dim,
                kernel_size=1,
                stride=1,
                ),
            )
        
    def forward(self, x):
        x = x + self.pre_norm_attn(x)
        x = x + self.pre_norm_ffn(x)
        return x

class MobileViTBlockv2(nn.Module):
    def __init__(
        self, 
        in_channels, 
        attn_unit_dim, 
        ffn_multiplier, 
        n_attn_blocks, 
        patch_h, 
        patch_w,
        ):
        super().__init__()
        self.in_channels = in_channels
        self.attn_unit_dim = attn_unit_dim
        self.ffn_multiplier = ffn_multiplier
        self.n_attn_blocks = n_attn_blocks
        self.patch_h = patch_h
        self.patch_w = patch_w
        
        self.local_rep = self._build_local_rep()
        self.global_rep = self._build_global_rep()
        self.conv_proj = nn.Sequential(
            nn.Conv2d(
                in_channels=self.attn_unit_dim,
                out_channels=self.in_channels,
                kernel_size=1,
                stride=1,
                ),
            nn.BatchNorm2d(self.in_channels),
            )
    
    def _build_local_rep(self):
        conv_3x3 = nn.Sequential(
            nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.in_channels,
                kernel_size=3,
                groups=self.in_channels,
                stride=1,
                padding=(1,1),
                padding_mode='zeros',
                ),
            nn.BatchNorm2d(self.in_channels),
            nn.SiLU(),
            )
        conv_1x1 = nn.Sequential(
            nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.attn_unit_dim,
                kernel_size=1,
                stride=1,
                ),
            )
        return nn.Sequential(conv_3x3, conv_1x1)
        
    def _build_global_rep(self):
        # TODO: Finish _build_attn_layer        
        ffn_dims = [self.ffn_multiplier * self.attn_unit_dim] * self.n_attn_blocks
        ffn_dims = [int((d//16)*16) for d in ffn_dims]
        global_rep = [
            LinearAttnFFN(
                embed_dim=self.attn_unit_dim,
                ffn_latent_dim=ffn_dim,
                )
            for ffn_dim in ffn_dims
            ]
        global_rep.append(
            nn.GroupNorm(
                num_groups=1, 
                num_channels=self.attn_unit_dim, 
                eps=1e-5, 
                affine=True,
                )
            )
        return nn.Sequential(*global_rep)
    
    def resize_input(self, x):
        b, c, h, w = x.shape
        if h%self.patch_h != 0 or w%self.patch_w != 0:
            h = int(math.ceil(h/self.patch_h)*self.patch_h)
            w = int(math.ceil(w/self.patch_w)*self.patch_w)
            x = F.interpolate(
                x, size=(h,w), mode='bilinear', align_corners=True,
                )
        return x
            
    def unfolding(self, x):
        b, c, h, w = x.shape
        patches = F.unfold(
            x,
            kernel_size=(self.patch_h, self.patch_w),
            stride=(self.patch_h, self.patch_w),
            )
        patches = patches.reshape(b,c,self.patch_h*self.patch_w,-1)
                
        return patches, (h,w)
    
    def folding(self, patches, output_size):
        b, c, p, n = patches.shape
        patches = patches.reshape(b, c*p, n)
        x = F.fold(
            patches,
            output_size=output_size,
            kernel_size=(self.patch_h, self.patch_w),
            stride=(self.patch_h, self.patch_w),
            )
        return x
    
    def forward(self, x):
        x = self.resize_input(x)
        fm = self.local_rep(x)
        patches, output_size = self.unfolding(fm)
        patches = self.global_rep(patches)
        fm = self.folding(patches=patches, output_size=output_size)
        fm = self.conv_proj(fm)
        return fm

class MobileViTv2(nn.Module):
    """Code rewriten from apple/ml-cvnets"""
    def __init__(self, in_channels=3, width_multiplier=0.75, num_classes=136):
        super().__init__()
        
        # --- Stem ---
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=int(32*width_multiplier),
                kernel_size=3,
                stride=2,
                padding=(1,1),
                padding_mode='zeros',
                ),
            nn.BatchNorm2d(int(32*width_multiplier)),
            nn.SiLU(),
            )
        self.out_channels = int(32*width_multiplier)
        
        # --- CONV layers ---
        self.layer1 = self._make_layer(
            block_type='mv2',
            block_config={
                'out_channels': int(64*width_multiplier),
                'expand_ratio': 2,
                'num_blocks': 1,
                'stride': 1,
                },
            )
        self.layer2 = self._make_layer(
            block_type='mv2',
            block_config={
                'out_channels': int(128*width_multiplier),
                'expand_ratio': 2,
                'num_blocks': 2,
                'stride': 2,
                },
            )
        self.layer3 = self._make_layer(
            block_type='mobilevit',
            block_config={
                'out_channels': int(256*width_multiplier),
                'attn_unit_dim': int(128*width_multiplier),
                'ffn_multiplier': 2,
                'attn_blocks': 2,
                'patch_h': 2,
                'patch_w': 2,
                'stride': 2,
                'mv_expand_ratio': 2,
                },
            )
        self.layer4 = self._make_layer(
            block_type='mobilevit',
            block_config={
                'out_channels': int(384*width_multiplier),
                'attn_unit_dim': int(192*width_multiplier),
                'ffn_multiplier': 2,
                'attn_blocks': 4,
                'patch_h': 2,
                'patch_w': 2,
                'stride': 2,
                'mv_expand_ratio': 2,
                },
            )
        self.layer5 = self._make_layer(
            block_type='mobilevit',
            block_config={
                'out_channels': int(512*width_multiplier),
                'attn_unit_dim': int(256*width_multiplier),
                'ffn_multiplier': 2,
                'attn_blocks': 3,
                'patch_h': 2,
                'patch_w': 2,
                'stride': 2,
                'mv_expand_ratio': 2,
                },
            )
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.classifier = nn.Linear(self.out_channels, num_classes, bias=True)
        
        self.reset_params()
        
    def calculate_parameters(self):
        return sum(p.numel() for p in self.parameters())
        
    def reset_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.weight is not None:
                    nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.ones_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                if m.weight is not None:
                    nn.init.trunc_normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def _make_layer(self, block_type, block_config):
        if block_type == "mobilevit":
            return self._make_mobilevit_layer(config=block_config)
        elif block_type == "mv2":
            return self._make_mv2_layer(config=block_config)
            
    def _make_mobilevit_layer(self, config):
        stride = config['stride']
        
        block = []
        if stride == 2:
            layer = InvertedResidual(
                in_channels=self.out_channels,
                out_channels=config['out_channels'],
                stride=2,
                expand_ratio=config['mv_expand_ratio'],
                )
            block.append(layer)
            self.out_channels = config['out_channels']
        block.append(
            MobileViTBlockv2(
                in_channels=self.out_channels,
                attn_unit_dim=config['attn_unit_dim'],
                ffn_multiplier=config['ffn_multiplier'],
                n_attn_blocks=config['attn_blocks'],
                patch_h=config['patch_h'],
                patch_w=config['patch_w'],
                )
            )
        return nn.Sequential(*block)
            
    def _make_mv2_layer(self, config):
        out_channels = config['out_channels']
        expand_ratio = config['expand_ratio']
        num_blocks = config['num_blocks']
        
        block = []
        for i in range(num_blocks):
            stride = config['stride'] if i==0 else 1
            layer = InvertedResidual(
                in_channels=self.out_channels,
                out_channels=out_channels,
                stride=stride,
                expand_ratio=expand_ratio,
                )
            block.append(layer)
            self.out_channels = out_channels
        return nn.Sequential(*block)
    
    def extract_features(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.pool(x)
        x = x.flatten(start_dim=1)
        return x
    
    def forward(self, x):
        x = self.extract_features(x)
        x = self.classifier(x)
        return x