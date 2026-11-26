"""Shared two-view detector; only local backbone P3 receives interaction."""
import torch
from torch import nn
from geometry import align_global_p3, alignment_grid


class P3Interaction(nn.Module):
    def __init__(self, channels=256, width=64):
        super().__init__()
        self.reduce = nn.Conv2d(channels * 2, width, 1, bias=False)
        self.spatial = nn.Conv2d(width, width, 3, padding=1, groups=width, bias=False)
        self.act = nn.SiLU()
        self.restore = nn.Conv2d(width, channels, 1, bias=False)
        nn.init.zeros_(self.restore.weight)

    def forward(self, local, context, valid):
        residual = self.restore(self.act(self.spatial(self.act(self.reduce(torch.cat((local, context), 1))))))
        return local + residual * valid.to(local.dtype)


class DualViewDEIM(nn.Module):
    def __init__(self, backbone, encoder, decoder, mode='aligned', width=64):
        super().__init__()
        if mode not in ('independent', 'aligned', 'local_residual'):
            raise ValueError(mode)
        self.backbone, self.encoder, self.decoder = backbone, encoder, decoder
        self.mode = mode
        # Common initialization contains the same dormant module in all arms.
        # Independent mode never executes it and has fewer active parameters.
        self.interaction = P3Interaction(256, width)
        if mode == 'independent':
            self.interaction.requires_grad_(False)

    def interact(self, local_features, global_features, crop_norm):
        features = list(local_features)
        if self.mode == 'independent':
            return features
        if self.mode == 'aligned':
            context, valid = align_global_p3(global_features[0], crop_norm, features[0].shape[-2:])
        else:
            # Match both context and output masking of the aligned arm. Masking
            # only the output allows padded context to leak through the 3x3 DW.
            _, valid = alignment_grid(crop_norm.float(), features[0].shape[-2:])
            context = features[0] * valid.to(features[0].dtype)
        features[0] = self.interaction(features[0], context, valid)
        return features

    def forward(self, samples, targets=None):
        full, local, crop = samples['global'], samples['local'], samples['crop_norm']
        if full.shape != local.shape or full.ndim != 4 or crop.shape != (full.shape[0], 4):
            raise ValueError('Expected paired [B,3,640,640] and [B,4] crop coordinates')
        batch = full.shape[0]
        # Single combined call controls shared-backbone BN statistics in both arms.
        features = self.backbone(torch.cat((full, local), 0))
        global_features = [x[:batch] for x in features]
        local_features = self.interact([x[batch:] for x in features], global_features, crop)
        encoded = self.encoder([torch.cat((g, l), 0) for g, l in zip(global_features, local_features)])
        global_targets = targets['global'] if targets else None
        local_targets = targets['local'] if targets else None
        # Separate native decoder calls preserve its per-view DN/loss contracts.
        return {'global': self.decoder([x[:batch] for x in encoded], global_targets),
                'local': self.decoder([x[batch:] for x in encoded], local_targets)}

    def cache_global(self, image):
        if self.training:
            raise RuntimeError('Caching is evaluation-only')
        feats = self.backbone(image)
        return feats, self.decoder(self.encoder(feats))

    def predict_local(self, image, crop_norm, global_features):
        if self.training:
            raise RuntimeError('Use paired forward for training')
        local = self.interact(self.backbone(image), global_features, crop_norm)
        return self.decoder(self.encoder(local))


def paired_loss(criterion, outputs, targets):
    # One original-image pair is one optimization sample; losses average two views.
    a = criterion(outputs['global'], targets['global'])
    b = criterion(outputs['local'], targets['local'])
    # Empty crops legitimately omit denoising losses in the native decoder.
    # Missing view-specific terms contribute zero; do not fabricate GT or DN.
    return {k: .5 * (a.get(k, 0.) + b.get(k, 0.)) for k in a.keys() | b.keys()}

