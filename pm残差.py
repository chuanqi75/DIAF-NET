import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import functools
import warnings
from thop import profile  # For FLOPs and Params calculation
import argparse
from torchvision.models import resnet18, ResNet18_Weights

# Mock CBAM (replace with actual import if available)
try:
    from models.CBAM import CBAM
except ImportError:
    class CBAM(nn.Module):
        def __init__(self, channel, reduction=16):
            super(CBAM, self).__init__()
            self.channel = channel
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
            self.max_pool = nn.AdaptiveMaxPool2d(1)
            self.fc1 = nn.Conv2d(channel, channel // reduction, 1, bias=False)
            self.relu = nn.ReLU(inplace=True)
            self.fc2 = nn.Conv2d(channel // reduction, channel, 1, bias=False)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
            max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
            out = self.sigmoid(avg_out + max_out)
            return x * out


# Custom ResNet18 to return intermediate features
class ResNet18Features(nn.Module):
    def __init__(self, pretrained=True):
        super(ResNet18Features, self).__init__()
        # Load pretrained ResNet18
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = resnet18(weights=weights)

        # Extract layers
        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x0 = self.layer1(x)  # [B, 64, H/4, W/4]
        x1 = self.layer2(x0)  # [B, 128, H/8, W/8]
        x2 = self.layer3(x1)  # [B, 256, H/16, W/16]
        x3 = self.layer4(x2)  # [B, 512, H/32, W/32]

        return x0, x1, x2, x3


# Mock resnet module
class resnet:
    @staticmethod
    def resnet18(pretrained=True):
        return ResNet18Features(pretrained=pretrained)


# Utility Classes
class TwoLayerConv2d(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__(nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                   padding=kernel_size // 2, stride=1, bias=False),
                         nn.BatchNorm2d(in_channels),
                         nn.ReLU(),
                         nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                                   padding=kernel_size // 2, stride=1))


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


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channels // ratio, in_channels, 1, bias=False)
        self.sigmod = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmod(out)


class DMlp(nn.Module):
    def __init__(self, dim, growth_rate=2.0):
        super().__init__()
        hidden_dim = int(dim * growth_rate)
        self.conv_0 = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 3, 1, 1, groups=dim),
            nn.Conv2d(hidden_dim, hidden_dim, 1, 1, 0)
        )
        self.act = nn.GELU()
        self.conv_1 = nn.Conv2d(hidden_dim, dim, 1, 1, 0)

    def forward(self, x):
        x = self.conv_0(x)
        x = self.act(x)
        x = self.conv_1(x)
        return x


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


class SupervisedAttentionModule(nn.Module):
    def __init__(self, mid_d):
        super(SupervisedAttentionModule, self).__init__()
        self.mid_d = mid_d
        self.cbam = CBAM(channel=self.mid_d)
        self.conv2 = nn.Sequential(
            nn.Conv2d(self.mid_d, self.mid_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        context = self.cbam(x)
        x_out = self.conv2(context)
        return x_out


class EnhancedSAM(nn.Module):
    def __init__(self, dim):
        super(EnhancedSAM, self).__init__()
        self.sam = SupervisedAttentionModule(dim)
        self.smfa = SMFA(dim=dim)

    def forward(self, x):
        attended = self.sam(x)
        enhanced = self.smfa(attended)
        return enhanced


# Decoder Definitions
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


class UNetDecoder(nn.Module):
    def __init__(self, skip_channels, decoder_channels):
        super(UNetDecoder, self).__init__()
        self.blocks = nn.ModuleList()
        for i in range(len(decoder_channels)):
            if i == 0:
                in_ch = skip_channels[-1]
            else:
                in_ch = decoder_channels[i - 1]
            skip_ch = skip_channels[-(i + 2)] if i < len(skip_channels) - 1 else 0
            out_ch = decoder_channels[i]
            self.blocks.append(UNetDecoderBlock(in_ch, skip_ch, out_ch))

    def forward(self, features):
        features = features[::-1]
        x = features[0]
        skips = features[1:]
        for i, block in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            x = block(x, skip)
        return x


class UNetDecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super(UNetDecoderBlock, self).__init__()
        if skip_channels > 0:
            self.concat_channels = in_channels + skip_channels
        else:
            self.concat_channels = in_channels
        self.upsampling = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv1 = nn.Conv2d(self.concat_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x, skip=None):
        x = self.upsampling(x)
        if skip is not None:
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, skip], dim=1)
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        return x


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


# Backbone and SEIFNet
class Backbone(nn.Module):
    def __init__(self, args, input_nc, output_nc,
                 resnet_stages_num=5,
                 output_sigmoid=False, if_upsample_2x=True):
        super(Backbone, self).__init__()
        self.upsamplex2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.upsamplex4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.classifier = TwoLayerConv2d(in_channels=32, out_channels=output_nc)
        self.if_upsample_2x = if_upsample_2x
        self.output_sigmoid = output_sigmoid
        self.sigmoid = nn.Sigmoid()

    def forward_single(self, x):
        f = self.backbone(x)
        return f


class SEIFNet(Backbone):
    def __init__(self, args, input_nc, output_nc,
                 decoder_type='DecoderBlock', decoder_softmax=False, embed_dim=64,
                 Building_Bool=False):
        super(SEIFNet, self).__init__(args, input_nc, output_nc)
        self.stage_dims = [64, 128, 256, 512]
        self.output_nc = output_nc
        self.backbone = resnet.resnet18(pretrained=True)

        self.diff1 = ImprovedCoDEM2(self.stage_dims[0])
        self.diff2 = ImprovedCoDEM2(self.stage_dims[1])
        self.diff3 = ImprovedCoDEM2(self.stage_dims[2])
        self.diff4 = ImprovedCoDEM2(self.stage_dims[3])

        if decoder_type == 'DecoderBlock':
            self.decoder_3 = DecoderBlock(self.stage_dims[3], self.stage_dims[2], self.stage_dims[2])
            self.decoder_2 = DecoderBlock(self.stage_dims[2], self.stage_dims[1], self.stage_dims[1])
            self.decoder_1 = DecoderBlock(self.stage_dims[1], self.stage_dims[0], self.stage_dims[0])
        elif decoder_type == 'UNetDecoder':
            skip_channels = self.stage_dims[::-1]  # [512, 256, 128, 64]
            decoder_channels = [256, 128, 64]
            self.decoder = UNetDecoder(skip_channels, decoder_channels)
        elif decoder_type == 'PANDecoder':
            self.decoder_3 = PANDecoder(self.stage_dims[3], self.stage_dims[2], self.stage_dims[2])
            self.decoder_2 = PANDecoder(self.stage_dims[2], self.stage_dims[1], self.stage_dims[1])
            self.decoder_1 = PANDecoder(self.stage_dims[1], self.stage_dims[0], self.stage_dims[0])
        else:
            raise ValueError(f"Unknown decoder_type: {decoder_type}")

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

        if hasattr(self, 'decoder'):
            # UNetDecoder path
            features = [d4, d3, d2, d1]
            p1 = self.decoder(features)
            out = self._upsample_like(self.final_conv(p1), x1)
        else:
            # DecoderBlock or PANDecoder path
            p3 = self.decoder_3(d4, d3)
            p2 = self.decoder_2(p3, d2)
            p1 = self.decoder_1(p2, d1)
            out = self._upsample_like(self.final_conv(p1), x1)

        return out


# Helper Functions
def init_weights(net, init_type='normal', init_gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            nn.init.normal_(m.weight.data, 1.0, init_gain)
            nn.init.constant_(m.bias.data, 0.0)

    net.apply(init_func)


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    if len(gpu_ids) > 0:
        assert torch.cuda.is_available()
        net.to(gpu_ids[0])
        if len(gpu_ids) > 1:
            net = torch.nn.DataParallel(net, gpu_ids)
    init_weights(net, init_type, init_gain=init_gain)
    return net


# FLOPs and Params Calculation
def calculate_flops_params():
    # Mock args for SEIFNet
    class Args:
        net_G = 'SEIFNet'
        embed_dim = 64

    args = Args()

    # Input tensors
    input_x1 = torch.randn(1, 3, 256, 256)
    input_x2 = torch.randn(1, 3, 256, 256)

    # Decoder types to evaluate
    decoder_types = ['DecoderBlock', 'UNetDecoder', 'PANDecoder']

    # Format output
    def format_flops(flops):
        if flops >= 1e9:
            return f"{flops / 1e9:.2f} GFLOPs"
        elif flops >= 1e6:
            return f"{flops / 1e6:.2f} MFLOPs"
        else:
            return f"{flops:.2f} FLOPs"

    def format_params(params):
        if params >= 1e6:
            return f"{params / 1e6:.2f} M"
        elif params >= 1e3:
            return f"{params / 1e3:.2f} K"
        else:
            return f"{params:.2f}"

    # Calculate for each decoder type
    for decoder_type in decoder_types:
        print(f"\nCalculating for SEIFNet with {decoder_type}...")
        model = SEIFNet(args, input_nc=3, output_nc=2, decoder_type=decoder_type)
        model.eval()
        flops, params = profile(model, inputs=(input_x1, input_x2), verbose=False)
        print(f"Model: SEIFNet ({decoder_type})")
        print(f"Input Size: [1, 3, 256, 256] x 2")
        print(f"FLOPs: {format_flops(flops)}")
        print(f"Params: {format_params(params)}")


if __name__ == "__main__":
    calculate_flops_params()