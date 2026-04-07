from dictdot import dictdot
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import torchvision
from torchvision import models as torchvision_models

from omegaconf import OmegaConf
import math
import os
import INN
import INN.INNAbstract as INNAbstract
from INN.CouplingModels.conv import CouplingConv
from INN.CouplingModels.NICEModel.conv import Conv2dNICE
from INN.CouplingModels.utils import _default_2d_coupling_function

class ConvNICEfast(CouplingConv):
    '''
    1-d invertible convolution layer by NICE method
    '''
    def __init__(self, channels, kernel_size, w=4, activation_fn=nn.ReLU, m=None, mask=None):
        super(ConvNICEfast, self).__init__(num_feature=channels, mask=mask)
    
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        x2 = x2 + self.m1(x1)
        x1 = x1 + self.m2(x2)
        return torch.cat([x1, x2], dim=1)
    
    def inverse(self, y):
        y1, y2 = y.chunk(2, dim=1)
        y1 = y1 - self.m2(y2)
        y2 = y2 - self.m1(y1)        
        return torch.cat([y1, y2], dim=1)    

    def logdet(self, **args):
        return 0


class Conv2dNICEFast(ConvNICEfast):
    '''
    1-d invertible convolution layer by NICE method
    '''
    def __init__(self, channels, kernel_size, w=4, activation_fn=nn.ReLU, m=None, mask=None):
        super(Conv2dNICEFast, self).__init__(channels, kernel_size, w=w, activation_fn=activation_fn, m=m, mask=mask)
        self.kernel_size = kernel_size
        self.m1 = _default_2d_coupling_function(channels // 2, kernel_size, activation_fn, w=w)
        self.m2 = _default_2d_coupling_function(channels // 2, kernel_size, activation_fn, w=w)

    def forward(self, x, log_p0=0, log_det_J=0):
        y = super(Conv2dNICEFast, self).forward(x)
        if self.compute_p:
            return y, log_p0, log_det_J + self.logdet()
        else:
            return y
    
    def inverse(self, y, **args):
        x = super(Conv2dNICEFast, self).inverse(y)
        return x
    
    def __repr__(self):
        return f'Conv2dNICE(channels={self.num_feature}, kernel_size={self.kernel_size})'


def unwrap_model(model):
    return model.module if hasattr(model, 'module') else model

class PSNR:
    """Peak Signal to Noise Ratio
    img1 and img2 have range [0, 255]"""

    def __init__(self):
        self.name = "PSNR"

    @staticmethod
    def __call__(img1, img2, dim=[1, 2, 3]):
        mse = torch.mean((img1 - img2) ** 2, dim=dim)
        return 20 * torch.log10(255.0 / torch.sqrt(mse))


def get_psnr(x_input, x_recon, zero_mean=False, is_video=False):
    if zero_mean:
        x_input_0_255 = (x_input + 1) * 127.5
        x_recon_0_255 = (x_recon + 1) * 127.5
    else:
        x_input_0_255 = x_input * 255
        x_recon_0_255 = x_recon * 255
    if is_video:
        psnr = PSNR()(x_input_0_255, x_recon_0_255, dim=[1, 2, 3, 4])
    else:
        psnr = PSNR()(x_input_0_255, x_recon_0_255)
    return psnr


def nonlinearity(x):
    # swish
    return x * torch.sigmoid(x)


def Normalize(in_channels, num_groups=32):
    return torch.nn.GroupNorm(
        num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True
    )


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=1, padding=1
            )

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=2, padding=0
            )

    def forward(self, x):
        if self.with_conv:
            pad = (0, 1, 0, 1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class ResnetBlock(nn.Module):
    def __init__(
        self,
        *,
        in_channels,
        out_channels=None,
        conv_shortcut=False,
        dropout,
        temb_channels=512,
    ):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels, out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(
                    in_channels, out_channels, kernel_size=3, stride=1, padding=1
                )
            else:
                self.nin_shortcut = torch.nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=1, padding=0
                )

    def _forward(self, x, temb):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x + h

    def forward(self, x, temb):
        return checkpoint(self._forward, x, temb, use_reentrant=False)


class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.k = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.v = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.proj_out = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q.shape
        q = q.reshape(b, c, h * w)
        q = q.permute(0, 2, 1)  # b,hw,c
        k = k.reshape(b, c, h * w)  # b,c,hw
        w_ = torch.bmm(q, k)  # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c) ** (-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b, c, h * w)
        w_ = w_.permute(0, 2, 1)  # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v, w_)  # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b, c, h, w)

        h_ = self.proj_out(h_)

        return x + h_
class Decoder(nn.Module):
    def __init__(
        self,
        *,
        ch=128,
        out_ch=3,
        ch_mult=(1, 1, 2, 2, 4),
        num_res_blocks=2,
        attn_resolutions=(16,),
        dropout=0.0,
        resamp_with_conv=True,
        in_channels=3,
        resolution=256,
        z_channels=16,
        give_pre_end=False,
        mid_attn=True,
        **ignore_kwargs,
    ):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end

        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1,) + tuple(ch_mult)
        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        self.z_shape = (1, z_channels, curr_res, curr_res)
        # print(
        #     "Working with z of shape {} = {} dimensions.".format(
        #         self.z_shape, np.prod(self.z_shape)
        #     )
        # )

        # z to block_in
        self.conv_in = torch.nn.Conv2d(
            z_channels, block_in, kernel_size=3, stride=1, padding=1
        )

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )
        self.mid_attn = mid_attn
        if self.mid_attn:
            self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                block.append(
                    ResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        temb_channels=self.temb_ch,
                        dropout=dropout,
                    )
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(
            block_in, out_ch, kernel_size=3, stride=1, padding=1
        )

    def forward(self, z):
        # assert z.shape[1:] == self.z_shape[1:]
        self.last_z_shape = z.shape

        # timestep embedding
        temb = None

        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h, temb)
        if self.mid_attn:
            h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h, temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        if self.give_pre_end:
            return h

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h

class PixelUnshuffle2d(INNAbstract.PixelShuffleModule):
    def __init__(self, r):
        super(PixelUnshuffle2d, self).__init__()
        self.r = r
        self.shuffle = nn.PixelShuffle(r)
        self.unshuffle = nn.PixelUnshuffle(r)
    
    def PixelShuffle(self, x):
        return self.unshuffle(x)
    
    def PixelUnshuffle(self, x):
        return self.shuffle(x)

class PIXELVAEfast(nn.Module):
    def __init__(self, use_flow=True, *args, **kwargs):
        super().__init__()

        self.flow = INN.Sequential(
            Conv2dNICE(3, 3), # 16
            Conv2dNICE(3, 3), # 16
            INN.PixelShuffle2d(2), # 8
            Conv2dNICEFast(12, 3),
            Conv2dNICEFast(12, 3),
            INN.PixelShuffle2d(2), # 4
            Conv2dNICEFast(48, 3),
            Conv2dNICEFast(48, 3),
            INN.PixelShuffle2d(2), # 2
            Conv2dNICEFast(192, 3),
            Conv2dNICEFast(192, 3),
            INN.PixelShuffle2d(2), # 1
            Conv2dNICEFast(768, 3, w=1),
            Conv2dNICEFast(768, 3, w=1),
        )
        self.ps = INN.PixelShuffle2d(2)
        self.flow.computing_p(True)
        self.flow_lam = 1
        self.decoder = Decoder(out_ch=3, ch_mult=[1, 1, 1, 1, 1], num_res_blocks=1, z_channels=768, mid_attn=False)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(3072, 1000)

    def encode(self, x, *args, **kwargs):            
        z = self.flow(x)[0]
        # z = F.pixel_unshuffle(x, 2)
        return z

    def decode(self, z, *args, **kwargs):
        # x = F.pixel_shuffle(z, 2)
        x = self.flow.inverse(z)
        return dictdot(dict(sample=x))

    def forward(self, x, is_student):
        z = self.encode(x)
        if is_student:
            xhat = self.decoder(z)
        else:
            xhat = None
        z = self.avgpool(self.ps(z)[0])
        z = torch.flatten(z, 1)
        z = self.fc(z)
        return z, xhat

    def get_last_layer(self):
        return self.flow[13].m1.f[4].weight


class PIXELVAE(nn.Module):
    def __init__(self, use_flow=True, *args, **kwargs):
        super().__init__()

        self.flow = INN.Sequential(
            Conv2dNICE(3, 3), # 16
            Conv2dNICE(3, 3), # 16
            INN.PixelShuffle2d(2), # 8
            Conv2dNICE(12, 3),
            Conv2dNICE(12, 3),
            INN.PixelShuffle2d(2), # 4
            Conv2dNICE(48, 3),
            Conv2dNICE(48, 3),
            INN.PixelShuffle2d(2), # 2
            Conv2dNICE(192, 3),
            Conv2dNICE(192, 3),
            INN.PixelShuffle2d(2), # 1
            Conv2dNICE(768, 3, w=1),
            Conv2dNICE(768, 3, w=1),
        )
        self.ps = INN.PixelShuffle2d(2)
        self.flow.computing_p(True)
        self.flow_lam = 1
        self.decoder = Decoder(out_ch=3, ch_mult=[1, 1, 1, 1, 1], num_res_blocks=1, z_channels=768, mid_attn=False)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(3072, 1000)

    def encode(self, x, *args, **kwargs):            
        z = self.flow(x)[0]
        # z = F.pixel_unshuffle(x, 2)
        return z

    def decode(self, z, *args, **kwargs):
        # x = F.pixel_shuffle(z, 2)
        x = self.flow.inverse(z)
        return dictdot(dict(sample=x))

    def forward(self, x, is_student):
        z = self.encode(x)
        if is_student:
            xhat = self.decoder(z)
        else:
            xhat = None
        z = self.avgpool(self.ps(z)[0])
        z = torch.flatten(z, 1)
        z = self.fc(z)
        return z, xhat

    def get_last_layer(self):
        return self.flow[13].m.f[4].weight


if __name__ == "__main__":
    vae = PIXELVAEfast().cuda()
    x = torch.randn([4,3,256,256]).cuda()
    z = vae.encode(x)
    xhat = vae.decode(z).sample
    print(torch.mean((x - xhat)**2))
