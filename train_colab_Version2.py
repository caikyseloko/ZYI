#!/usr/bin/env python3
"""
Script de treino preparado para Colab:
- Use com GPU (Runtime -> GPU)
- Ajuste paths para o diretório onde guardou data/coco128
- Salva checkpoints no checkpoint-dir (pode apontar para /content/drive/MyDrive/...)

Exemplo:
python train_colab.py --data-dir data/coco128 --epochs 10 --batch-size 8 --checkpoint-dir /content/drive/MyDrive/zyi_ckpts
"""
import os
import argparse
import math
import torch
from torch.utils.data import DataLoader
from zyi.dataset import COCO128Dataset
from zyi.utils import Vocab
from zyi.models import SimpleTextEncoder, Composer, FlowPatch, ImplicitPainter
import torch.nn.functional as F

def collate_fn(batch, vocab, max_len=32):
    imgs, caps = zip(*batch)
    imgs = torch.stack(imgs, dim=0)
    toks = torch.stack([vocab.encode(c, max_len=max_len) for c in caps], dim=0)
    return imgs, toks

def build_vocab_if_needed(data_dir, sample_size=5000):
    vocab_path = os.path.join(data_dir, 'vocab.json')
    if os.path.exists(vocab_path):
        return Vocab.load(vocab_path)
    ds_tmp = COCO128Dataset(data_dir, max_items=sample_size)
    all_caps = [ds_tmp.ann[k] for k in ds_tmp.filenames]
    v = Vocab()
    v.build_from_corpus(all_caps, min_freq=2, max_size=10000)
    v.save(vocab_path)
    return v

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    vocab = build_vocab_if_needed(args.data_dir)
    dataset = COCO128Dataset(args.data_dir, max_items=args.max_items)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2,
                        collate_fn=lambda b: collate_fn(b, vocab, max_len=32))

    text_enc = SimpleTextEncoder(vocab_size=len(vocab.itos), d_model=384).to(device)
    composer = Composer(d_text=384, latent_ch=48).to(device)
    # define patching: split image 128x128 into 8x8 patches of 16x16 -> P=64 patches
    P = (128 // 16) ** 2
    patch_dim = 48 * 1  # dimension per-patch vector (we will pool)
    flow = FlowPatch(patch_dim=patch_dim, cond_dim=384, n_layers=6).to(device)
    painter = ImplicitPainter(cond_dim=384 + patch_dim, img_size=128).to(device)

    optim = torch.optim.Adam(list(text_enc.parameters()) + list(composer.parameters()) + list(flow.parameters()) + list(painter.parameters()), lr=args.lr)

    for epoch in range(args.epochs):
        running_loss = 0.0
        for i, (imgs, toks) in enumerate(loader):
            imgs = imgs.to(device)
            toks = toks.to(device)
            B = imgs.size(0)
            text_emb = text_enc(toks)  # (B, D)
            z_comp = torch.randn(B, 64, device=device)
            latents = composer(text_emb, z_comp)
            top = latents[-1]  # (B,C,128,128)

            # build patch vectors: adaptive pool to (B, Patches, patch_dim)
            # here choose pooling to 16x16 patches -> pool to 8x8 then flatten per patch
            # Simpler approach: downsample top to (B, C, 8,8) then each spatial location is a patch vector
            patch_map = F.adaptive_avg_pool2d(top, (8,8))  # (B, C, 8,8)
            B, C, H, W = patch_map.shape
            patches = patch_map.view(B, C, H*W).permute(0,2,1)  # (B, P, C)
            # if needed pad/truncate to patch_dim
            if patches.size(2) > patch_dim:
                patches = patches[:, :, :patch_dim]
            elif patches.size(2) < patch_dim:
                pad = torch.zeros(B, patches.size(1), patch_dim - patches.size(2), device=patches.device)
                patches = torch.cat([patches, pad], dim=2)

            # flow forward: patches (x) -> z with logdet
            z, logdet = flow.forward(patches, text_emb)
            # negative loglikelihood under standard normal: 0.5 * ||z||^2 - logdet
            nll = 0.5 * (z ** 2).sum(dim=(1,2)) - logdet  # sum over patch dims and patches
            nll = nll.mean()

            # build painter conditioning vector per image: concat text_emb + global pool of top + pooled patch mean
            global_pool = top.view(B, C, -1).mean(dim=2)
            patch_mean = patches.view(B, -1).mean(dim=1, keepdim=True)  # (B,1) not ideal but simple
            # create final cond per image by concatenation (match ImplicitPainter cond_dim)
            # ImplicitPainter expects cond_dim = 384 + patch_dim in our setup; create cond by merging text_emb + flattened small vector
            # For simplicity we use text_emb + mean of patches repeated/padded to match patch_dim
            patch_summary = patches.view(B, -1)[:, :patch_dim]  # (B, patch_dim)
            cond = torch.cat([text_emb, patch_summary.mean(dim=1, keepdim=True).expand(-1, patch_dim)], dim=1)
            # cond currently shape (B, 384+patch_dim) because concatenation above sets that (works with earlier comment)

            # generate image
            gen = painter(cond)

            # losses
            pixel_loss = F.mse_loss(gen, imgs)
            loss = args.lambda_flow * nll + args.lambda_pixel * pixel_loss

            optim.zero_grad()
            loss.backward()
            optim.step()

            running_loss += loss.item()
            if (i+1) % args.log_interval == 0:
                avg = running_loss / args.log_interval
                print(f"Epoch {epoch} iter {i+1}/{len(loader)} loss {avg:.4f}")
                running_loss = 0.0

        # checkpoint
        ckpt = {
            'text_enc': text_enc.state_dict(),
            'composer': composer.state_dict(),
            'flow': flow.state_dict(),
            'painter': painter.state_dict(),
            'vocab': len(vocab.itos)
        }
        torch.save(ckpt, os.path.join(args.checkpoint_dir, f'epoch_{epoch}.pth'))
        print("Saved checkpoint epoch", epoch)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', required=True)
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--checkpoint-dir', default='checkpoints')
    p.add_argument('--max-items', type=int, default=20000)
    p.add_argument('--log-interval', type=int, default=50)
    p.add_argument('--lambda-flow', type=float, default=1.0)
    p.add_argument('--lambda-pixel', type=float, default=1.0)
    args = p.parse_args()
    main(args)