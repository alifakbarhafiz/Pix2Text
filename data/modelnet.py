import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from pix2text.data.utils import load_off, normalize_pc, batched_fps


def get_classes(modelnet_root):
    dirs = [d for d in os.listdir(modelnet_root) if os.path.isdir(os.path.join(modelnet_root, d))]
    return sorted([d for d in dirs if not d.startswith('.')])


def build_npz_fast(root_dir, out_file, classes, points_per_model=1024, val_ratio=0.1, device='cuda', batch_size=64):
    root_dir = Path(root_dir)
    all_points, all_labels = [], []
    file_list = []
    for cls_idx, cls in enumerate(classes):
        for split in ["train","test"]:
            off_dir = root_dir / cls / split
            if not off_dir.exists():
                continue
            for fname in sorted(os.listdir(off_dir)):
                if fname.endswith(".off"):
                    file_list.append((cls_idx, off_dir / fname))

    print(f"Total OFF files: {len(file_list)}")
    pbar = tqdm(total=len(file_list), desc="Building NPZ", unit="file")

    for i in range(0, len(file_list), batch_size):
        batch_files = file_list[i:i+batch_size]
        pcs, labels = [], []

        for cls_idx, fpath in batch_files:
            try:
                v = normalize_pc(load_off(fpath))
                if v.shape[0] >= points_per_model:
                    v_tensor = torch.tensor(v, dtype=torch.float32).unsqueeze(0).to(device)
                    sampled = batched_fps(v_tensor, points_per_model)[0].cpu().numpy()
                else:
                    idx = np.random.choice(v.shape[0], points_per_model, replace=True)
                    sampled = v[idx]
                pcs.append(torch.tensor(sampled, dtype=torch.float32))
                labels.append(cls_idx)
            except Exception as e:
                print("Failed:", fpath, e)
                continue

        if len(pcs) == 0:
            pbar.update(len(batch_files))
            continue

        pcs = torch.stack(pcs)
        all_points.append(pcs.numpy())
        all_labels.extend(labels)
        pbar.update(len(batch_files))

    pbar.close()
    if len(all_points) == 0:
        raise RuntimeError("No valid meshes found.")

    all_points = np.concatenate(all_points)
    all_labels = np.array(all_labels)

    idx = np.arange(len(all_points))
    np.random.shuffle(idx)
    train_n = int(len(all_points)*(1-val_ratio))

    np.savez(out_file,
             train_points=all_points[idx[:train_n]],
             train_labels=all_labels[idx[:train_n]],
             val_points=all_points[idx[train_n:]],
             val_labels=all_labels[idx[train_n:]])
    print(f"NPZ created: {out_file}, total models: {len(all_points)}")


class ModelNetNPZ(Dataset):
    def __init__(self, pts, labels, use_num_points=1024):
        self.pts = pts
        self.labels = labels
        self.use_num_points = use_num_points

    def __len__(self):
        return len(self.pts)

    def __getitem__(self, i):
        pc = self.pts[i]
        idx = np.random.choice(pc.shape[0], self.use_num_points, replace=False)
        pc = pc[idx]
        return torch.tensor(pc, dtype=torch.float32), torch.tensor(self.labels[i], dtype=torch.long)
