# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 19:41:55 2022

@author: AustinHsu
"""

import torch
import torch.nn as nn

class WingLoss(nn.Module):
    
    def __init__(self, omega=10, epsilon=2):
        super().__init__()
        self.omega = omega
        self.epsilon = epsilon
        self.const_c = omega - omega * ( torch.Tensor([omega/epsilon]) + 1 ).log()
        
    def forward(self, pred, target):
        diff = (target - pred).abs()
        y1 = diff[diff<self.omega]
        y2 = diff[diff>=self.omega]
        
        loss_below = self.omega * (1+y1/self.epsilon).log()
        loss_above = y2 - self.const_c.type_as(y2)
        
        return (loss_below.sum() + loss_above.sum()) / pred.shape[0]
    
class SmoothL1Loss(nn.Module):
    
    def __init__(self, threshold = 1):
        super().__init__()
        self.threshold = threshold
        
    def forward(self, pred, target):
        diff = (target - pred).abs()
        y1 = diff[diff<self.threshold]
        y2 = diff[diff>=self.threshold]
        
        loss_below = 0.5 * y1**2
        loss_above = y2 - 0.5
        return (loss_below.sum() + loss_above.sum()) / pred.shape[0]
    
class AdpativeWingLoss(nn.Module):
    
    def __init__(self, omega=14, theta=0.5, epsilon=1, alpha=2.1):
        super().__init__()
        self.omega = omega
        self.theta = theta
        self.epsilon = epsilon
        self.alpha = alpha
        
    def forward(self, pred, target):
        diff = (target - pred).abs()
        diff_y1 = diff[diff<self.theta]
        diff_y2 = diff[diff>=self.theta]
        y1 = target[diff<self.theta]
        y2 = target[diff>=self.theta]
        
        loss_below = self.omega * (1+ torch.pow(diff_y1/self.epsilon, self.alpha-y1) ).log()
        temp_y2 = self.alpha - y2
        theta_div = self.theta/self.epsilon
        A = self.omega * (1/self.epsilon) * temp_y2 * ( torch.pow(theta_div, temp_y2-1) / ( 1+torch.pow(theta_div, temp_y2) ) )
        C = self.theta * A - self.omega * (1+torch.pow(theta_div, temp_y2)).log()
        loss_above = A * diff_y2 - C
        
        return (loss_below.sum() + loss_above.sum()) / pred.shape[0]
    
class GaussianNegativeLogLikelihoodLoss(nn.Module):
    
    def __init__(self, num_keypoints: int = 68):
        super().__init__()
        self.num_keypoints = num_keypoints
        self.keypoint_weight = nn.Parameter(torch.Tensor([[10. for _ in range(num_keypoints)]]), requires_grad=True)
        
    def forward(self, pred, target, log_sigmas):
        pred_landmark = pred.view(-1,self.num_keypoints,2)
        target_landmark = target.view(-1,self.num_keypoints,2)
        loss_mu = (pred_landmark - target_landmark).abs().sum(dim=-1) # (b,kps)
        loss_mu /= 2*torch.pow(log_sigmas.exp(),2) # (b,kps)
        loss_sigma = 2*log_sigmas.abs() # (b,kps)
        return ( nn.functional.sigmoid(self.keypoint_weight) * (loss_sigma + loss_mu) ).mean(dim=-1).mean()