from functools import partial

import torch
import torch.nn as nn

from timm.models.vision_transformer import PatchEmbed, Block

from util.pos_embed import get_2d_sincos_pos_embed


class AxialAttentionBlock(nn.Module):
    """Cross-Attention: center slice tokens (Q) attend to neighbor slice tokens (KV).

    Applies after a ViT block in the last axial_depth layers of the encoder.
    Residual connection + FFN keeps it as a lightweight plug-in.
    """

    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm_q  = norm_layer(embed_dim)
        self.norm_kv = norm_layer(embed_dim)
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm2 = norm_layer(embed_dim)
        mlp_hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, embed_dim),
        )

    def forward(self, x_center, x_neighbors):
        """
        x_center   : (B, L_c, D)  — center slice visible tokens
        x_neighbors: (B, L_n, D)  — all neighbor slice tokens concatenated
        """
        q  = self.norm_q(x_center)
        kv = self.norm_kv(x_neighbors)
        attn_out, _ = self.cross_attn(q, kv, kv)
        x_center = x_center + attn_out
        x_center = x_center + self.mlp(self.norm2(x_center))
        return x_center


class SpatialAxialDMAE(nn.Module):
    """Spatial-Axial Denoising Masked Autoencoder (SA-DMAE).

    Extends DMAE to 2.5D by processing n_slices consecutive slices:
      - Spatial Stream : center slice with Gaussian noise + random masking (DMAE as-is)
      - Axial Stream   : all slices without masking, shared ViT encoder weights
      - Fusion         : AxialAttentionBlock after each of the last axial_depth ViT blocks
      - Decoder        : identical to DMAE, reconstructs the center slice only
    """

    def __init__(
        self,
        img_size=224, patch_size=16, in_chans=3,
        embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4.0, norm_layer=nn.LayerNorm, norm_pix_loss=False,
        sigma=0.5,
        n_slices=3, axial_depth=4,
    ):
        super().__init__()

        assert axial_depth <= depth, "axial_depth must be <= encoder depth"
        assert n_slices % 2 == 1, "n_slices must be odd so center index is well-defined"

        self.sigma       = sigma
        self.n_slices    = n_slices
        self.axial_depth = axial_depth
        self.norm_pix_loss = norm_pix_loss

        # ── Encoder ──────────────────────────────────────────────────────────
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False
        )

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for _ in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        # Axial Attention plug-ins (one per last axial_depth ViT blocks)
        self.axial_blocks = nn.ModuleList([
            AxialAttentionBlock(embed_dim, num_heads, mlp_ratio, norm_layer)
            for _ in range(axial_depth)
        ])

        # ── Decoder (identical to DMAE) ──────────────────────────────────────
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False
        )
        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for _ in range(decoder_depth)
        ])
        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size ** 2 * in_chans, bias=True)

        self.initialize_weights()

    # ── Weight initialisation ────────────────────────────────────────────────

    def initialize_weights(self):
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1], int(self.patch_embed.num_patches ** 0.5), cls_token=True
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        dec_pos_embed = get_2d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches ** 0.5), cls_token=True
        )
        self.decoder_pos_embed.data.copy_(torch.from_numpy(dec_pos_embed).float().unsqueeze(0))

        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        torch.nn.init.normal_(self.cls_token, std=0.02)
        torch.nn.init.normal_(self.mask_token, std=0.02)
        self.apply(self._init_weights)

        # ImageNet normalisation constants (updated to device in forward)
        self.mean = torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
        self.std  = torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    # ── Patch utilities (operate on center slice only) ───────────────────────

    def patchify(self, imgs):
        """(N, C, H, W) → (N, L, patch**2 * C)"""
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0
        h = w = imgs.shape[2] // p
        c = imgs.shape[1]
        x = imgs.reshape(imgs.shape[0], c, h, p, w, p)
        x = torch.einsum('nchpwq->nhwpqc', x)
        return x.reshape(imgs.shape[0], h * w, p ** 2 * c)

    def unpatchify(self, x):
        """(N, L, patch**2 * C) → (N, C, H, W)"""
        p = self.patch_embed.patch_size[0]
        c = self.patch_embed.proj.out_channels   # embed projection out = in_chans handled elsewhere
        # infer in_chans from pred size
        in_chans = x.shape[-1] // (p ** 2)
        h = w = int(x.shape[1] ** 0.5)
        x = x.reshape(x.shape[0], h, w, p, p, in_chans)
        x = torch.einsum('nhwpqc->nchpwq', x)
        return x.reshape(x.shape[0], in_chans, h * p, h * p)

    def random_masking(self, x, mask_ratio):
        """Per-sample random masking via argsort noise (identical to DMAE)."""
        N, L, D = x.shape
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore  = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return x_masked, mask, ids_restore

    # ── Encoder ──────────────────────────────────────────────────────────────

    def _embed_slice(self, x):
        """Patch-embed one slice and add positional encoding (no cls token yet)."""
        x = self.patch_embed(x)           # (B, L, D)
        x = x + self.pos_embed[:, 1:, :]  # skip cls pos
        return x

    def forward_encoder(self, x_slices, mask_ratio):
        """
        x_slices : (B, n_slices, C, H, W) — already normalised
        Returns  : (latent, mask, ids_restore)
                   latent shape: (B, L_visible + 1, embed_dim)  [+1 for cls]
        """
        B, N, C, H, W = x_slices.shape
        center_idx = N // 2

        device = x_slices.device
        cls_token = self.cls_token + self.pos_embed[:, :1, :]  # (1, 1, D)

        # ── Spatial Stream: center slice with noise + masking ─────────────────
        x_center = x_slices[:, center_idx]              # (B, C, H, W)
        noise    = torch.randn_like(x_center) * self.sigma
        x_center = x_center + noise

        x = self._embed_slice(x_center)                 # (B, L, D)
        x, mask, ids_restore = self.random_masking(x, mask_ratio)
        x = torch.cat([cls_token.expand(B, -1, -1), x], dim=1)  # (B, L_vis+1, D)

        # ── Axial Stream: each neighbor slice (no masking, shared weights) ────
        neighbor_feats = []
        for i in range(N):
            if i == center_idx:
                continue
            xi = self._embed_slice(x_slices[:, i])      # (B, L, D)
            xi = torch.cat([cls_token.expand(B, -1, -1), xi], dim=1)  # (B, L+1, D)
            neighbor_feats.append(xi)
        # neighbor_feats: list of (B, L+1, D), length = n_slices - 1

        # ── ViT blocks ────────────────────────────────────────────────────────
        spatial_blocks = self.blocks[: -self.axial_depth]
        late_blocks    = self.blocks[-self.axial_depth :]

        # Early blocks: spatial only (neighbors processed with shared weights)
        for blk in spatial_blocks:
            x = blk(x)
            neighbor_feats = [blk(xn) for xn in neighbor_feats]

        # Late blocks: ViT block → Axial Attention (interleaved)
        for blk, axial_blk in zip(late_blocks, self.axial_blocks):
            x = blk(x)
            neighbor_feats = [blk(xn) for xn in neighbor_feats]

            x_neigh = torch.cat(neighbor_feats, dim=1)  # (B, (N-1)*(L+1), D)
            x = axial_blk(x, x_neigh)                   # Cross-Attention fusion

        x = self.norm(x)
        return x, mask, ids_restore

    # ── Decoder (identical to DMAE) ──────────────────────────────────────────

    def forward_decoder(self, x, ids_restore):
        x = self.decoder_embed(x)

        mask_tokens = self.mask_token.repeat(
            x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1
        )
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
        x  = torch.cat([x[:, :1, :], x_], dim=1)

        x = x + self.decoder_pos_embed

        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        x = self.decoder_pred(x)
        x = x[:, 1:, :]  # remove cls token
        return x

    # ── Loss (operates on center slice only) ─────────────────────────────────

    def forward_loss(self, imgs_center, pred, mask):
        """
        imgs_center: (B, C, H, W) — clean center slice, normalised
        pred       : (B, L, p*p*C)
        mask       : (B, L)  0=keep, 1=masked
        """
        target = self.patchify(imgs_center)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var  = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1e-6) ** 0.5

        loss = (pred - target) ** 2
        loss = loss.mean()
        return loss

    # ── Forward ──────────────────────────────────────────────────────────────

    def forward(self, x_slices, mask_ratio=0.75):
        """
        x_slices: (B, n_slices, C, H, W)
                  Channel dim can be 1 (grayscale) or 3 (RGB).
        Returns : (loss, pred, mask)
        """
        B, N, C, H, W = x_slices.shape

        # Move normalisation tensors to correct device
        if self.mean.device != x_slices.device:
            self.mean = self.mean.to(x_slices.device)
            self.std  = self.std.to(x_slices.device)

        # Normalise (broadcast over slice dim)
        mean = self.mean
        std  = self.std
        if C == 1:
            # Grayscale: use single-channel stats (mean of RGB mean/std)
            mean = mean.mean(dim=1, keepdim=True)
            std  = std.mean(dim=1, keepdim=True)

        x_norm = (x_slices - mean.unsqueeze(1)) / std.unsqueeze(1)  # (B, N, C, H, W)

        center_idx    = N // 2
        x_center_norm = x_norm[:, center_idx]                        # (B, C, H, W)

        latent, mask, ids_restore = self.forward_encoder(x_norm, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)
        loss = self.forward_loss(x_center_norm, pred, mask)
        return loss, pred, mask


# ── Model factories ──────────────────────────────────────────────────────────

def sa_dmae_vit_base_patch16(**kwargs):
    return SpatialAxialDMAE(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )


def sa_dmae_vit_large_patch16(**kwargs):
    return SpatialAxialDMAE(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
