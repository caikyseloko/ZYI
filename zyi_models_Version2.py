"""
ZYI-Net core modules (nova arquitetura):

- SimpleTextEncoder: pequeno Transformer encoder treinado do zero
- Composer: gera pirâmide de latentes multiescala
- FlowPatch: normalizing flow (RealNVP-like) que atua por patch para coerência local
- ImplicitPainter: MLP implícito vetorizado que gera pixels a partir de coordenadas e condicionamento por texto+latent
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# --------- Text Encoder ----------
class SimpleTextEncoder(nn.Module):
    def __init__(self, vocab_size, d_model=384, nhead=6, nlayers=4, max_len=32):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos = nn.Embedding(max_len, d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model*4)
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=nlayers)
        self.d_model = d_model
        self.max_len = max_len

    def forward(self, toks):
        # toks: (B, T)
        B, T = toks.shape
        positions = torch.arange(0, T, device=toks.device).unsqueeze(0).expand(B, T)
        x = self.tok(toks) + self.pos(positions)
        x = x.permute(1,0,2)  # (T,B,D)
        out = self.enc(x)
        out = out.permute(1,0,2)  # (B,T,D)
        return out.mean(dim=1)  # (B,D)

# --------- Composer (multiscale) ----------
class Composer(nn.Module):
    def __init__(self, d_text=384, latent_ch=48):
        super().__init__()
        self.fc = nn.Linear(d_text + 64, 8*8*latent_ch)
        self.latent_ch = latent_ch

    def forward(self, text_embed, z):
        x = torch.cat([text_embed, z], dim=-1)
        x = self.fc(x)
        x = x.view(-1, self.latent_ch, 8, 8)
        l1 = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)   # 16
        l2 = F.interpolate(l1, scale_factor=2, mode='bilinear')  # 32
        l3 = F.interpolate(l2, scale_factor=2, mode='bilinear')  # 64
        l4 = F.interpolate(l3, scale_factor=2, mode='bilinear')  # 128
        return [x, l1, l2, l3, l4]

# --------- FlowPatch: RealNVP-like for patch vectors ----------
class AffineCoupling(nn.Module):
    def __init__(self, dim, cond_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim//2 + cond_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, dim//2 * 2)
        )
    def forward(self, x, cond):
        x1, x2 = x.chunk(2, dim=1)
        h = torch.cat([x1, cond], dim=1)
        st = self.net(h)
        s, t = st.chunk(2, dim=1)
        s = torch.tanh(s) * 0.5
        y2 = x2 * torch.exp(s) + t
        y = torch.cat([x1, y2], dim=1)
        logdet = s.sum(dim=1)
        return y, logdet

    def inverse(self, y, cond):
        y1, y2 = y.chunk(2, dim=1)
        h = torch.cat([y1, cond], dim=1)
        st = self.net(h)
        s, t = st.chunk(2, dim=1)
        s = torch.tanh(s) * 0.5
        x2 = (y2 - t) * torch.exp(-s)
        x = torch.cat([y1, x2], dim=1)
        logdet = -s.sum(dim=1)
        return x, logdet

class FlowPatch(nn.Module):
    def __init__(self, patch_dim, cond_dim, n_layers=6):
        super().__init__()
        self.layers = nn.ModuleList([AffineCoupling(patch_dim, cond_dim) for _ in range(n_layers)])
        self.patch_dim = patch_dim

    def forward(self, x, cond):
        # x: (B, P, D) patches flattened; cond: (B, cond_dim)
        B, P, D = x.shape
        logdet = x.new_zeros(B)
        z = x
        for i, l in enumerate(self.layers):
            # process per patch
            z_new = []
            ld_sum = z.new_zeros(B)
            for p in range(P):
                zp, ldet = l(z[:,p,:], cond)
                z_new.append(zp.unsqueeze(1))
                ld_sum = ld_sum + ldet
            z = torch.cat(z_new, dim=1)
            # simple channel permutation: reverse dims
            z = z.flip(dims=[2])
            logdet = logdet + ld_sum
        return z, logdet

    def inverse(self, z, cond):
        B, P, D = z.shape
        logdet = z.new_zeros(B)
        x = z
        for l in reversed(self.layers):
            x = x.flip(dims=[2])
            x_new = []
            ld_sum = x.new_zeros(B)
            for p in range(P):
                xp, ldet = l.inverse(x[:,p,:], cond)
                x_new.append(xp.unsqueeze(1))
                ld_sum = ld_sum + ldet
            x = torch.cat(x_new, dim=1)
            logdet = logdet + ld_sum
        return x, logdet

# --------- ImplicitPainter: vectorized MLP that maps coords + cond -> RGB ----------
class ImplicitMLP(nn.Module):
    def __init__(self, cond_dim, hidden=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 + cond_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 3), nn.Tanh()
        )
    def forward(self, coords, cond):
        # coords: (N,2), cond: (B,cond_dim) or (cond_dim,)
        if coords.dim() != 2:
            raise ValueError("coords must be (N,2)")
        N = coords.size(0)
        if cond.dim() == 1:
            cond_exp = cond.unsqueeze(0).expand(N, -1)
        else:
            cond_exp = cond.expand(N, -1)
        inp = torch.cat([coords, cond_exp], dim=1)
        return self.net(inp)

class ImplicitPainter(nn.Module):
    def __init__(self, cond_dim, img_size=128):
        super().__init__()
        self.implicit = ImplicitMLP(cond_dim)
        self.img_size = img_size
        # precompute coords grid in normalized [-1,1]
        xs = torch.linspace(-1,1,img_size)
        yy, xx = torch.meshgrid(xs, xs, indexing='ij')
        self.register_buffer('coords_grid', torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1))  # (H*W,2)

    def forward(self, cond):
        # cond: (B, cond_dim)
        B = cond.size(0)
        N = self.coords_grid.size(0)
        coords = self.coords_grid
        out = []
        for b in range(B):
            rgb_flat = self.implicit(coords, cond[b])  # (N,3)
            img = rgb_flat.permute(1,0).view(3, self.img_size, self.img_size)
            out.append(img.unsqueeze(0))
        return torch.cat(out, dim=0)  # (B,3,H,W)