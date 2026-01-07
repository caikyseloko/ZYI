#!/usr/bin/env python3
"""
Inferência simples: gera imagem a partir de texto usando checkpoint.
Exemplo:
python eval.py --checkpoint checkpoints/epoch_0.pth --vocab data/coco128/vocab.json --text "um cão na praia" --out gen.png
"""
import argparse
import torch
from zyi.utils import Vocab
from zyi.models import SimpleTextEncoder, Composer, FlowPatch, ImplicitPainter
from PIL import Image
import numpy as np

def load_models(checkpoint, vocab_path, device='cuda'):
    vocab = Vocab.load(vocab_path)
    ckpt = torch.load(checkpoint, map_location=device)
    text_enc = SimpleTextEncoder(vocab_size=len(vocab.itos), d_model=384).to(device)
    composer = Composer(d_text=384, latent_ch=48).to(device)
    patch_dim = 48 * 1
    flow = FlowPatch(patch_dim=patch_dim, cond_dim=384, n_layers=6).to(device)
    painter = ImplicitPainter(cond_dim=384 + patch_dim, img_size=128).to(device)

    text_enc.load_state_dict(ckpt['text_enc'])
    composer.load_state_dict(ckpt['composer'])
    flow.load_state_dict(ckpt['flow'])
    painter.load_state_dict(ckpt['painter'])
    return text_enc, composer, flow, painter, vocab

def generate(checkpoint, vocab_path, text, device='cuda'):
    text_enc, composer, flow, painter, vocab = load_models(checkpoint, vocab_path, device)
    device = torch.device(device)
    text_ids = vocab.encode(text, max_len=32).unsqueeze(0).to(device)
    with torch.no_grad():
        text_emb = text_enc(text_ids)
        z_comp = torch.randn(1,64, device=device)
        latents = composer(text_emb, z_comp)
        top = latents[-1]
        # build patch summary as in train (simple)
        patch_map = F.adaptive_avg_pool2d(top, (8,8))
        B, C, H, W = patch_map.shape
        patches = patch_map.view(1, C, H*W).permute(0,2,1)
        if patches.size(2) < 48:
            pad = torch.zeros(1, patches.size(1), 48 - patches.size(2), device=device)
            patches = torch.cat([patches, pad], dim=2)
        patch_summary = patches.view(1, -1)[:, :48]
        cond = torch.cat([text_emb, patch_summary.mean(dim=1, keepdim=True).expand(-1, 48)], dim=1)
        gen = painter(cond)
        img = ((gen[0].clamp(-1,1) + 1)/2 * 255).cpu().numpy().astype('uint8')
        img = img.transpose(1,2,0)
        return Image.fromarray(img)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--vocab', required=True)
    p.add_argument('--text', required=True)
    p.add_argument('--out', default='gen.png')
    p.add_argument('--device', default='cuda')
    args = p.parse_args()
    pil = generate(args.checkpoint, args.vocab, args.text, device=args.device)
    pil.save(args.out)
    print("Saved", args.out)