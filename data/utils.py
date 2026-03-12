import os
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


def load_off(file_path):
    """Load vertices from an OFF file. Copes with normal OFF, concatenated header (OFF123 456), or no OFF line."""
    with open(file_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]

    first_line = lines[0].upper()
    if first_line.startswith('OFF'):
        if first_line == 'OFF':
            header_line = lines[1]
            lines = lines[1:]
        else:
            header_line = first_line[3:].strip()
            lines[0] = header_line
    else:
        header_line = first_line

    parts = header_line.split()
    num_vertices = int(parts[0])
    vertices = np.array([list(map(float, lines[1 + j].split())) for j in range(num_vertices)], dtype=np.float32)
    return vertices


def normalize_pc(pc):
    pc = pc - np.mean(pc, axis=0, keepdims=True)
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    return pc / m


def batched_fps(points, k):
    B, N, _ = points.shape
    device = points.device
    sampled_idx = torch.zeros((B, k), dtype=torch.long, device=device)
    distances = torch.full((B, N), float('inf'), device=device)
    farthest = torch.randint(0, N, (B,), device=device)
    batch_idx = torch.arange(B, device=device)
    for i in range(k):
        sampled_idx[:, i] = farthest
        centroid = points[batch_idx, farthest].unsqueeze(1)
        dist = torch.sum((points - centroid) ** 2, dim=2)
        distances = torch.minimum(distances, dist)
        farthest = torch.argmax(distances, dim=1)
    return points[batch_idx.unsqueeze(1), sampled_idx]
