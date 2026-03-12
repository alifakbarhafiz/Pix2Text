import os
import torch

from pix2text.training.loss import info_nce
from pix2text import config


def extract_epoch(fname):
    try:
        return int(fname.split('_')[1].split('.')[0])
    except Exception:
        return -1


def run_training(
    pc_encoder,
    txt_encoder,
    pc_proj,
    txt_proj,
    opt,
    train_points,
    train_labels,
    val_points,
    val_labels,
    CLASS_DESC,
    device,
    SAVE_DIR=None,
    BATCH_SIZE=128,
    NUM_EPOCHS=200,
):
    if SAVE_DIR is None:
        SAVE_DIR = config.SAVE_DIR
    os.makedirs(SAVE_DIR, exist_ok=True)

    start_epoch = 1
    best_val_loss = float("inf")

    # pick up from last ckpt if any
    ckpt_list = [f for f in os.listdir(SAVE_DIR)
                 if f.endswith(".pt") and f.startswith("ckpt_") and f != "best_ckpt.pt"]

    if ckpt_list:
        latest_ckpt = sorted(ckpt_list, key=extract_epoch)[-1]
        ckpt_path = os.path.join(SAVE_DIR, latest_ckpt)
        print("Resuming from checkpoint:", ckpt_path)

        ckpt = torch.load(ckpt_path, map_location=device)
        pc_encoder.load_state_dict(ckpt["pc_encoder"])
        txt_encoder.load_state_dict(ckpt["txt_encoder"])
        pc_proj.load_state_dict(ckpt["pc_proj"])
        txt_proj.load_state_dict(ckpt["txt_proj"])
        opt.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        if "best_val_loss" in ckpt:
            best_val_loss = ckpt["best_val_loss"]

    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        # Training
        pc_encoder.train()
        txt_encoder.train()
        pc_proj.train()
        txt_proj.train()
        total_loss = 0
        num_batches = 0

        for i in range(0, len(train_points), BATCH_SIZE):
            pc_batch = train_points[i:i+BATCH_SIZE].to(device)
            text_batch = [CLASS_DESC[l.item()] for l in train_labels[i:i+BATCH_SIZE]]

            pc_emb = pc_proj(pc_encoder(pc_batch))
            txt_emb = txt_proj(txt_encoder(text_batch))

            opt.zero_grad()
            loss = info_nce(pc_emb, txt_emb)
            loss.backward()
            opt.step()

            total_loss += loss.item()
            num_batches += 1

        avg_train_loss = total_loss / num_batches

        # Validation
        pc_encoder.eval()
        txt_encoder.eval()
        pc_proj.eval()
        txt_proj.eval()
        val_loss_total = 0
        val_batches = 0

        with torch.no_grad():
            for i in range(0, len(val_points), BATCH_SIZE):
                pc_batch = val_points[i:i+BATCH_SIZE].to(device)
                text_batch = [CLASS_DESC[l.item()] for l in val_labels[i:i+BATCH_SIZE]]

                pc_emb = pc_proj(pc_encoder(pc_batch))
                txt_emb = txt_proj(txt_encoder(text_batch))

                val_loss_total += info_nce(pc_emb, txt_emb).item()
                val_batches += 1

        avg_val_loss = val_loss_total / val_batches
        print(f"Epoch {epoch} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # save this epoch
        ckpt_path = os.path.join(SAVE_DIR, f"ckpt_{epoch}.pt")
        torch.save({
            "epoch": epoch,
            "pc_encoder": pc_encoder.state_dict(),
            "txt_encoder": txt_encoder.state_dict(),
            "pc_proj": pc_proj.state_dict(),
            "txt_proj": txt_proj.state_dict(),
            "optimizer": opt.state_dict(),
            "best_val_loss": best_val_loss
        }, ckpt_path)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_path = os.path.join(SAVE_DIR, "best_ckpt.pt")
            torch.save({
                "epoch": epoch,
                "pc_encoder": pc_encoder.state_dict(),
                "txt_encoder": txt_encoder.state_dict(),
                "pc_proj": pc_proj.state_dict(),
                "txt_proj": txt_proj.state_dict(),
                "optimizer": opt.state_dict(),
                "best_val_loss": best_val_loss
            }, best_path)
            print(f"Best checkpoint updated: {best_path}")
