import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import functools
from . import resnet

class TwoLayerConv2d(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__(nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                   padding=kernel_size // 2, stride=1, bias=False),
                         nn.BatchNorm2d(in_channels),
                         nn.ReLU(),
                         nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                                   padding=kernel_size // 2, stride=1)
                         )

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)
# class ImprovedCoDEM2(nn.Module):
#
#     def __init__(self, channel_dim):
#         super(ImprovedCoDEM2, self).__init__()
#         self.channel_dim = channel_dim
#
#         self.Conv3 = nn.Conv2d(in_channels=2 * self.channel_dim, out_channels=2 * self.channel_dim,
#                                kernel_size=3, stride=1, padding=1)
#         self.Conv1 = nn.Conv2d(in_channels=2 * self.channel_dim, out_channels=self.channel_dim,
#                                kernel_size=1, stride=1, padding=0)
#
#         self.BN1 = nn.BatchNorm2d(2 * self.channel_dim)
#         self.BN2 = nn.BatchNorm2d(self.channel_dim)
#         self.ReLU = nn.ReLU(inplace=True)

#         self.res_conv = nn.Conv2d(in_channels=2 * channel_dim, out_channels=channel_dim,
#                                   kernel_size=1, stride=1, padding=0)
#         self.res_bn = nn.BatchNorm2d(channel_dim)
#
#         self.diff_conv = nn.Sequential(
#             nn.Conv2d(in_channels=channel_dim, out_channels=channel_dim, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(channel_dim),
#             nn.ReLU(inplace=True)
#         )
#
#     def forward(self, x1, x2):
#         B, C, H, W = x1.shape
#
#         f_d = torch.abs(x1 - x2)  # B,C,H,W
#
#         f_c = torch.cat((x1, x2), dim=1)  # B,2C,H,W
#
#         residual = self.res_bn(self.res_conv(f_c))

#         z_c = self.ReLU(self.BN2(self.Conv1(self.ReLU(self.BN1(self.Conv3(f_c))))))
#
#         z_d = self.diff_conv(f_d)
#         out = z_d + z_c + residual
#
#         return out
class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mip = max(8, inp // reduction)
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        a_h = a_h.expand(-1, -1, h, w)
        a_w = a_w.expand(-1, -1, h, w)
        return a_w, a_h

class ImprovedCoDEM2(nn.Module):
    def __init__(self, channel_dim):
        super(ImprovedCoDEM2, self).__init__()
        self.channel_dim = channel_dim
        self.Conv3 = nn.Conv2d(in_channels=2 * self.channel_dim, out_channels=2 * self.channel_dim,
                               kernel_size=3, stride=1, padding=1)
        self.Conv1 = nn.Conv2d(in_channels=2 * self.channel_dim, out_channels=self.channel_dim,
                               kernel_size=1, stride=1, padding=0)
        self.BN1 = nn.BatchNorm2d(2 * self.channel_dim)
        self.BN2 = nn.BatchNorm2d(self.channel_dim)
        self.ReLU = nn.ReLU(inplace=True)
        self.coordAtt = CoordAtt(inp=channel_dim, oup=channel_dim, reduction=16)
        self.res_conv = nn.Conv2d(in_channels=2 * channel_dim, out_channels=channel_dim,
                                  kernel_size=1, stride=1, padding=0)
        self.res_bn = nn.BatchNorm2d(channel_dim)

    def forward(self, x1, x2):
        B, C, H, W = x1.shape
        f_d = torch.abs(x1 - x2)
        f_c = torch.cat((x1, x2), dim=1)
        residual = self.res_bn(self.res_conv(f_c))
        z_c = self.ReLU(self.BN2(self.Conv1(self.ReLU(self.BN1(self.Conv3(f_c))))))
        d_aw, d_ah = self.coordAtt(f_d)
        z_d = f_d * d_aw * d_ah
        out = z_d + z_c + residual
        return out

class DecoderBlock(nn.Module):
    def __init__(self, high_level_ch, low_level_ch, out_ch):
        super(DecoderBlock, self).__init__()
        self.conv_high = nn.Conv2d(high_level_ch, out_ch, kernel_size=1)
        self.bn_high = nn.BatchNorm2d(out_ch)
        self.conv_low = nn.Conv2d(low_level_ch, out_ch, kernel_size=1)
        self.bn_low = nn.BatchNorm2d(out_ch)
        self.conv_cat = nn.Sequential(
            nn.Conv2d(out_ch * 2, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        self.residual = nn.Sequential(
            nn.Conv2d(out_ch * 2, out_ch, kernel_size=1),
            nn.BatchNorm2d(out_ch)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, high_level_feat, low_level_feat):
        h = self.bn_high(self.conv_high(high_level_feat))
        h = F.interpolate(h, size=low_level_feat.shape[2:], mode='bilinear', align_corners=False)
        l = self.bn_low(self.conv_low(low_level_feat))
        cat_feat = torch.cat([h, l], dim=1)
        out = self.conv_cat(cat_feat)
        residual = self.residual(cat_feat)
        out = self.relu(out + residual)
        return out

class DIAFNet(nn.Module):
    def __init__(self, args, input_nc, output_nc,
                 decoder_softmax=False, embed_dim=64,
                 Building_Bool=False):
        super(DIAFNet, self).__init__()
        self.stage_dims = [64, 128, 256, 512]
        self.output_nc = output_nc
        self.backbone = resnet.resnet18(pretrained=True)
        self.diff1 = ImprovedCoDEM2(self.stage_dims[0])
        self.diff2 = ImprovedCoDEM2(self.stage_dims[1])
        self.diff3 = ImprovedCoDEM2(self.stage_dims[2])
        self.diff4 = ImprovedCoDEM2(self.stage_dims[3])
        self.decoder_3 = DecoderBlock(self.stage_dims[3], self.stage_dims[2], self.stage_dims[2])
        self.decoder_2 = DecoderBlock(self.stage_dims[2], self.stage_dims[1], self.stage_dims[1])
        self.decoder_1 = DecoderBlock(self.stage_dims[1], self.stage_dims[0], self.stage_dims[0])
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.upsample8 = nn.Upsample(scale_factor=8, mode='bilinear')
        self.final_conv = nn.Sequential(
            nn.Conv2d(self.stage_dims[0], 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, output_nc, kernel_size=1)
        )

    def _upsample_like(self, x, target):
        _, _, h, w = target.size()
        return F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)

    def forward(self, x1, x2):
        f1 = self.backbone(x1)
        f2 = self.backbone(x2)
        x1_0, x1_1, x1_2, x1_3 = f1
        x2_0, x2_1, x2_2, x2_3 = f2
        d1 = self.diff1(x1_0, x2_0)
        d2 = self.diff2(x1_1, x2_1)
        d3 = self.diff3(x1_2, x2_2)
        d4 = self.diff4(x1_3, x2_3)
        p3 = self.decoder_3(d4, d3)
        p2 = self.decoder_2(p3, d2)
        p1 = self.decoder_1(p2, d1)
        out = self._upsample_like(self.final_conv(p1), x1)
        return out
class PANDecoder(nn.Module):
    def __init__(self, high_level_ch, low_level_ch, out_ch):
        super(PANDecoder, self).__init__()
        self.conv_high = nn.Conv2d(high_level_ch, out_ch, kernel_size=1)
        self.bn_high = nn.BatchNorm2d(out_ch)
        self.conv_low = nn.Conv2d(low_level_ch, out_ch, kernel_size=1)
        self.bn_low = nn.BatchNorm2d(out_ch)
        self.conv_top_down = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.bn_top_down = nn.BatchNorm2d(out_ch)
        self.conv_bottom_up = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=2, padding=1)
        self.bn_bottom_up = nn.BatchNorm2d(out_ch)
        self.conv_refine = nn.Conv2d(out_ch * 2, out_ch, kernel_size=3, padding=1)
        self.bn_refine = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, high_level_feat, low_level_feat):
        h = self.bn_high(self.conv_high(high_level_feat))
        h = F.interpolate(h, size=low_level_feat.shape[2:], mode='bilinear', align_corners=False)
        l = self.bn_low(self.conv_low(low_level_feat))
        fused = h + l
        top_down = self.bn_top_down(self.conv_top_down(fused))
        top_down = self.relu(top_down)
        bu = self.bn_bottom_up(self.conv_bottom_up(top_down))
        bu = F.interpolate(bu, size=top_down.shape[2:], mode='bilinear', align_corners=False)
        refined = torch.cat([top_down, bu], dim=1)
        out = self.bn_refine(self.conv_refine(refined))
        out = self.relu(out)
        return out
class SMFA(nn.Module):
    def __init__(self, dim=36):
        super(SMFA, self).__init__()
        self.linear_0 = nn.Conv2d(dim, dim * 2, 1, 1, 0)
        self.linear_1 = nn.Conv2d(dim, dim, 1, 1, 0)
        self.linear_2 = nn.Conv2d(dim, dim, 1, 1, 0)
        self.lde = DMlp(dim, 2)
        self.dw_conv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)
        self.gelu = nn.GELU()
        self.down_scale = 8
        self.alpha = nn.Parameter(torch.ones((1, dim, 1, 1)))
        self.belt = nn.Parameter(torch.zeros((1, dim, 1, 1)))

    def forward(self, f):
        _, _, h, w = f.shape
        y, x = self.linear_0(f).chunk(2, dim=1)
        x_s = self.dw_conv(F.adaptive_max_pool2d(x, (h // self.down_scale, w // self.down_scale)))
        x_v = torch.var(x, dim=(-2, -1), keepdim=True)
        x_l = x * F.interpolate(self.gelu(self.linear_1(x_s * self.alpha + x_v * self.belt)), size=(h, w),
                                mode='nearest')
        y_d = self.lde(y)
        return self.linear_2(x_l + y_d)
