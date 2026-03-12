import numpy as np
import torch
import torch.nn.functional as F


def describe_pointcloud(
    pc_np,
    pc_encoder,
    pc_proj,
    txt_encoder,
    txt_proj,
    CLASS_NAMES,
    device,
    use_num_points=1024,
):
    """pc_np is (N,3). Returns (best_class_name, similarity)."""
    idx = np.random.choice(pc_np.shape[0], use_num_points, replace=False)
    pc_np = pc_np[idx]

    pc = torch.tensor(pc_np, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        pc_emb = pc_proj(pc_encoder(pc))
        pc_emb = F.normalize(pc_emb, dim=1)

    texts = [f"A 3D object from the class {c}." for c in CLASS_NAMES]
    with torch.no_grad():
        txt_emb = txt_proj(txt_encoder(texts))
        txt_emb = F.normalize(txt_emb, dim=1)

    sims = F.cosine_similarity(pc_emb, txt_emb).cpu().numpy().flatten()
    top_idx = sims.argmax()
    return CLASS_NAMES[top_idx], float(sims[top_idx])


def describe_pointcloud_topk(
    pc_tensor,
    pc_encoder,
    pc_proj,
    txt_encoder,
    txt_proj,
    CLASS_NAMES,
    device,
    use_num_points=1024,
    top_k=3,
):
    """Accepts tensor or numpy (N,3). Returns (top_classes, top_scores, top_descriptions, pc_sampled)."""
    if isinstance(pc_tensor, torch.Tensor):
        pc_np = pc_tensor.detach().cpu().numpy()
    else:
        pc_np = pc_tensor

    idx = np.random.choice(pc_np.shape[0], use_num_points, replace=False)
    pc_sampled = pc_np[idx]

    pc_tensor_sampled = torch.tensor(pc_sampled, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        pc_emb = pc_proj(pc_encoder(pc_tensor_sampled))
        pc_emb = F.normalize(pc_emb, dim=1)

    texts = [f"A 3D object from the class {c}." for c in CLASS_NAMES]
    with torch.no_grad():
        txt_emb = txt_proj(txt_encoder(texts))
        txt_emb = F.normalize(txt_emb, dim=1)

    sims = F.cosine_similarity(pc_emb, txt_emb).cpu().numpy().flatten()
    top_idx = sims.argsort()[-top_k:][::-1]
    top_classes = [CLASS_NAMES[i] for i in top_idx]
    top_scores = sims[top_idx]
    top_descriptions = [f"{c} (sim={s:.3f})" for c, s in zip(top_classes, top_scores)]

    return top_classes, top_scores, top_descriptions, pc_sampled
