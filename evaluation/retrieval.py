import torch.nn.functional as F


def retrieval_accuracy(pc_emb, txt_emb, labels, train_labels_ref, topk=(1, 5)):
    """Top-k accuracy: for each pc, match to nearest text by cosine sim. labels = pc labels, train_labels_ref = text side labels."""
    pc_norm = F.normalize(pc_emb, dim=1)
    txt_norm = F.normalize(txt_emb, dim=1)

    sim_matrix = pc_norm @ txt_norm.T

    topk_values, topk_indices = sim_matrix.topk(max(topk), dim=1)
    correct = labels.unsqueeze(1) == train_labels_ref[topk_indices]

    acc = {}
    for k in topk:
        acc[k] = correct[:, :k].any(dim=1).float().mean().item() * 100
    return acc
