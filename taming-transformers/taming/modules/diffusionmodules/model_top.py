# pytorch_diffusion + derived encoder decoder
import math
import torch
import torch.nn as nn
import numpy as np
from taming.modules.vqvae.quantize import VectorQuantizer2 as VectorQuantizer


def get_timestep_embedding(timesteps, embedding_dim):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models:
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0,1,0,0))
    return emb


def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels,
                                    4*out_channels,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1)
        self.pixelshuffle = torch.nn.PixelShuffle(2)

    def forward(self, x):
        x = self.conv(x)
        x = self.pixelshuffle(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels=None, with_conv=True):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv2d(in_channels,
                                        out_channels,
                                        kernel_size=3,
                                        stride=2,
                                        padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = (0,1,0,1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, dropout=0.0, out_channels=None, conv_shortcut=False):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv2d(in_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)

        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(out_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(in_channels,
                                                     out_channels,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1)
            else:
                self.nin_shortcut = torch.nn.Conv2d(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0)

    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x+h

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, num_res_blocks=2, dropout=0.0, conv_shortcut=False):
        super().__init__()
        blocks = []
        for i_block in range(num_res_blocks):
            blocks.append(ResnetBlock(in_channels=in_channels,
                                         out_channels=out_channels,
                                         conv_shortcut = conv_shortcut,
                                         dropout=dropout))
        self.blocks = nn.Sequential(*blocks)
   
    def forward(self, x):
        return self.blocks(x)


class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)


    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b,c,h,w = q.shape
        q = q.reshape(b,c,h*w)
        q = q.permute(0,2,1)   # b,hw,c
        k = k.reshape(b,c,h*w) # b,c,hw
        w_ = torch.bmm(q,k)     # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b,c,h*w)
        w_ = w_.permute(0,2,1)   # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v,w_)     # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b,c,h,w)

        h_ = self.proj_out(h_)

        return x+h_


class Model(nn.Module):
    def __init__(self, 
                 n_embed,
                 embed_dim,
                 resolution, ch=128, down_f=16,
                 in_channels=3, out_channels=3, ch_mult=(1,1,2,2,4), 
                 num_res_blocks=2,beta=0.25,
                 dropout=0.0, resamp_with_conv=True, legacy=True):
        super().__init__()
        self.ch = ch
        self.num_stages = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        # conv_in
        self.conv_in = torch.nn.Conv2d(in_channels,
                                       ch,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        # top 
        self.top = nn.ModuleList()
        self.top.encode_0 = BasicBlock(ch, ch*ch_mult[0], num_res_blocks, dropout)
        self.top.encode_1 = torch.nn.Sequential(
            Downsample(ch*ch_mult[0], ch*ch_mult[1]),
            BasicBlock(ch*ch_mult[1], ch*ch_mult[1], num_res_blocks, dropout)
        )
        self.top.encode_2 = torch.nn.Sequential(
            Downsample(ch*ch_mult[1], ch*ch_mult[2]),
            BasicBlock(ch*ch_mult[2], ch*ch_mult[2], num_res_blocks, dropout)
        )
        self.top.encode_3 = torch.nn.Sequential(
            Downsample(ch*ch_mult[2], ch*ch_mult[3]),
            BasicBlock(ch*ch_mult[3], ch*ch_mult[3], num_res_blocks, dropout)
        )
        if down_f==16:
            self.top.encode_4 = torch.nn.Sequential(
                Downsample(ch*ch_mult[3], ch*ch_mult[4]),
                BasicBlock(ch*ch_mult[4], ch*ch_mult[4], num_res_blocks, dropout)
            )
        elif down_f==8:
            self.top.encode_4 = torch.nn.Sequential(
                torch.nn.Conv2d(ch*ch_mult[3], ch*ch_mult[4],kernel_size=3,stride=1,padding=1),
                BasicBlock(ch*ch_mult[4], ch*ch_mult[4], num_res_blocks, dropout)
            )

        in_ch = ch*ch_mult[4]
        self.top.encode_attn_1 = torch.nn.Sequential(
                            AttnBlock(in_ch), 
                            ResnetBlock(in_channels=in_ch, out_channels=in_ch, dropout=dropout)
                            )
        self.top.encode_attn_2 = torch.nn.Sequential(
                            AttnBlock(in_ch), 
                            ResnetBlock(in_channels=in_ch, out_channels=in_ch, dropout=dropout)
                            )
        self.top.encode_out = torch.nn.Sequential(
                            Normalize(in_ch), 
                            torch.nn.Conv2d(in_ch, embed_dim,kernel_size=3,stride=1,padding=1)
                            )    
        
        self.top.quantize = VectorQuantizer(n_embed, embed_dim, beta=beta, legacy=True)

        self.top.decode_convin = torch.nn.Conv2d(embed_dim,
                                       in_ch,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)
        self.top.decode_atten_1 = torch.nn.Sequential(
                            AttnBlock(in_ch), 
                            ResnetBlock(in_channels=in_ch, out_channels=in_ch, dropout=dropout)
                            )
        self.top.decode_atten_2 = torch.nn.Sequential(
                            AttnBlock(in_ch), 
                            ResnetBlock(in_channels=in_ch, out_channels=in_ch, dropout=dropout)
                            )

        if down_f==16:
            self.top.decode_0 = torch.nn.Sequential(
                BasicBlock(ch*ch_mult[4], ch*ch_mult[4], num_res_blocks, dropout),
                Upsample(ch*ch_mult[4], ch*ch_mult[3])
            )
        elif down_f==8:
            self.top.decode_0 = torch.nn.Sequential(
                torch.nn.Conv2d(ch*ch_mult[4], ch*ch_mult[3],kernel_size=3,stride=1,padding=1),
                BasicBlock(ch*ch_mult[3], ch*ch_mult[3], num_res_blocks, dropout)
            )
        self.top.decode_1 = torch.nn.Sequential(
            BasicBlock(ch*ch_mult[3], ch*ch_mult[3], num_res_blocks, dropout),
            Upsample(ch*ch_mult[3], ch*ch_mult[2])
        )
        self.top.decode_2 = torch.nn.Sequential(
            BasicBlock(ch*ch_mult[2], ch*ch_mult[2], num_res_blocks, dropout),
            Upsample(ch*ch_mult[2], ch*ch_mult[1])
        )
        self.top.decode_3 = torch.nn.Sequential(
            BasicBlock(ch*ch_mult[1], ch*ch_mult[1], num_res_blocks, dropout),
            Upsample(ch*ch_mult[1], ch*ch_mult[0])
        )
        self.top.decode_4 = BasicBlock(ch*ch_mult[1], ch, num_res_blocks, dropout)

        
        # end
        self.norm_out = Normalize(ch)
        self.conv_out = torch.nn.Conv2d(ch,
                                       out_channels,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)


    def forward(self, x):
        x = self.conv_in(x)
        x = self.top.encode_0(x)

        x_down_1 = self.top.encode_1(x)
        x_down_2 = self.top.encode_2(x_down_1)
        x_down_3 = self.top.encode_3(x_down_2)
        x_down_4 = self.top.encode_4(x_down_3)
        x_atten1 = self.top.encode_attn_1(x_down_4)
        x_atten2 = self.top.encode_attn_2(x_atten1)
        x_encode = self.top.encode_out(x_atten2)

        x_q, emb_loss, _ = self.top.quantize(x_encode)

        x = self.top.decode_convin(x_q)
        x = self.top.decode_atten_1(x)
        x = self.top.decode_atten_2(x)
        x = self.top.decode_0(x)
        x = self.top.decode_1(x)
        x = self.top.decode_2(x)
        x = self.top.decode_3(x)
        x = self.top.decode_4(x)

        x = self.norm_out(x)
        x = nonlinearity(x)
        out = self.conv_out(x)
        return out, emb_loss