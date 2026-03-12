import torch
import torch.nn.functional as F


def info_nce(pc_emb, txt_emb, temperature=0.07):
    """Contrastive loss: pc_emb and txt_emb are [B, D]. Symmetric pc<->text."""
    pc_emb = F.normalize(pc_emb, dim=1)
    txt_emb = F.normalize(txt_emb, dim=1)

    logits = torch.matmul(pc_emb, txt_emb.T) / temperature
    labels = torch.arange(pc_emb.size(0), device=pc_emb.device)

    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.T, labels)
    return (loss_i2t + loss_t2i) / 2
