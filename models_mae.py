# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

# MAE的核心实现
#- `MaskedAutoencoderViT` : 主要的MAE模型类
# - 包含编码器、解码器和掩码机制
from functools import partial

import torch
import torch.nn as nn

from timm.models.vision_transformer import PatchEmbed, Block

from util.pos_embed import get_2d_sincos_pos_embed


class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)#用于将输入图像转换为patch embeddings
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)#实现了标准的Transformer编码器块
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True) # decoder to patch
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))#torch.Size([1, 197, 768])

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)#torch.Size([1, 197, 512])
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))#torch.Size([1, 197, 512]) 解码器通常使用较小的维度（512）

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d) 将patch embedding的卷积层权重按照线性层的方式初始化 这样做是因为patch embedding在功能上类似于线性投影
        w = self.patch_embed.proj.weight.data#torch.Size([768, 3, 16, 16])
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))#这样做是因为patch embedding在功能上类似于线性投影
#特殊Token初始化
        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)  timm库的截断正态分布（cutoff=2.0）实际上等效于普通正态分布
        torch.nn.init.normal_(self.cls_token, std=.02)#分类token，用于全局特征表示使用标准差为0.02的正态分布初始化
        torch.nn.init.normal_(self.mask_token, std=.02)#掩码token，用于替换被掩盖的patch

        # initialize nn.Linear and nn.LayerNorm#其他层的初始化
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):#torch.Size([64, 3, 224, 224])
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]#16
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p#14
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))#torch.Size([64, 3, 14, 16, 14, 16])
        x = torch.einsum('nchpwq->nhwpqc', x)#torch.Size([64, 14, 14, 16, 16, 3])
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))#torch.Size([64, 196, 768])
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs
# 核心技术实现
    def random_masking(self, x, mask_ratio):#torch.Size([64, 196, 768]) 0.75
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        # 随机掩盖patches
    # 使用随机噪声排序来实现per-sample的随机掩码
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))#保留的patch数量 49
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1] torch.Size([64, 196])  
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove torch.Size([64, 196])
        ids_restore = torch.argsort(ids_shuffle, dim=1)#torch.Size([64, 196]) 用于恢复原始顺序

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]#torch.Size([64, 49])
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))#torch.Size([64, 49, 768])

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)#torch.Size([64, 196])
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio):#torch.Size([64, 3, 224, 224]) 0.75
        # embed patches
        #实际就是通过一个 nn.conv2d实现的，只需要 stride =patch 的大小就行了
        x = self.patch_embed(x)#用于将输入图像转换为patch embeddings 将2D图像转换为1D的patch序列 这是Vision Transformer处理图像的第一步。

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]#torch.Size([64, 196, 768]) batch_size, seq_len, hid_dim

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio)#torch.Size([64, 49, 768])  torch.Size([64, 196]) torch.Size([64, 196])

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]#torch.Size([1, 1, 768])
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)#torch.Size([64, 1, 768])
        x = torch.cat((cls_tokens, x), dim=1)#torch.Size([64, 49+1=50, 768])

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)#torch.Size([64, 50, 768])

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):#torch.Size([64, 50, 768]) torch.Size([64, 196])
        # embed tokens
        x = self.decoder_embed(x)#nn.Linear(embed_dim, decoder_embed_dim, bias=True) torch.Size([64, 50, 512])

        # append mask tokens to sequence self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)#torch.Size([64, 147, 512]) 添加Mask Tokens
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token torch.Size([64, 196, 512])
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle # 使用ids_restore将patches恢复到原始空间位置
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token # 重新添加CLS token

        # add pos embed添加位置编码
        x = x + self.decoder_pos_embed

        # apply Transformer blocksTransformer解码
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)#torch.Size([64, 197, 768])

        # predictor projection
        x = self.decoder_pred(x)#nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True)

        # remove cls token
        x = x[:, 1:, :]#torch.Size([64, 196, 768])

        return x

    def forward_loss(self, imgs, pred, mask):#torch.Size([64, 3, 224, 224]) torch.Size([64, 196, 768]) torch.Size([64, 196])
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        target = self.patchify(imgs)#将图像转换为patch序列 可以这么玩？
        if self.norm_pix_loss:#False
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, imgs, mask_ratio=0.75):
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)#torch.Size([64, 50, 768])  torch.Size([64, 196]) torch.Size([64, 196]) 
        pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask


def mae_vit_base_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_large_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_huge_patch14_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


# set recommended archs
mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_large_patch16 = mae_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_huge_patch14 = mae_vit_huge_patch14_dec512d8b  # decoder: 512 dim, 8 blocks
