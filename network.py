import torch
import torch.nn as nn
from torch import einsum
from einops import rearrange
from functools import partial
import torch.nn.functional as F
from timesformer.models.vit import TimeSformer
from i3d.model.I3D import InceptionI3d
from torchvision.ops import roi_align
import math
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '1, 0'

class TransFuse(nn.Module):
    def __init__(self, crops_pretrained_model, mouth_pretrained_model, num_frames=16, num_patches=14, embed_dim=768,
                 num_classes=2):
        super(TransFuse, self).__init__()

        self.timesformer = TimeSformer(img_size=224, num_classes=num_classes, num_frames=16,
                                       attention_type='divided_space_time',
                                       pretrained_model=str(crops_pretrained_model))
        self.i3d = InceptionI3d(num_classes=400, in_channels=3)
        self.i3d.load_state_dict(torch.load(mouth_pretrained_model))
        self.i3d.replace_logits(num_classes=num_classes)

        self.num_classes = num_classes
        self.num_frames = num_frames
        self.num_patches = num_patches
        self.embed_dim = embed_dim
        self.fusion_channels = [256, 512, 1024]

        self.end_points_1 = self.i3d.VALID_ENDPOINTS[: 6]
        self.end_points_2 = self.i3d.VALID_ENDPOINTS[6: 11]
        self.end_points_3 = self.i3d.VALID_ENDPOINTS[11: -2]

        self.fusion_1 = ConvTransBlock([0, 1, 2], self.timesformer, self.end_points_1, self.i3d, self.num_frames,
                                       self.num_patches, self.fusion_channels[0], 2, fusion=True, up_stride=2,
                                       down_stride=4)
        self.fusion_2 = ConvTransBlock([3, 4, 5], self.timesformer, self.end_points_2, self.i3d, self.num_frames,
                                       self.num_patches, self.fusion_channels[1], 4, fusion=True, up_stride=1,
                                       down_stride=2)
        self.fusion_3 = ConvTransBlock([6, 7, 8], self.timesformer, self.end_points_3, self.i3d, self.num_frames,
                                       self.num_patches, self.fusion_channels[2], 8, up_stride=0.5)
        self.fusion_4 = ConvTransBlock([9, 10, 11], self.timesformer, num_frames=self.num_frames,
                                       num_patches=self.num_patches)

        self.trans_norm = nn.LayerNorm(self.embed_dim)
        self.cls_head = nn.Linear(in_features=self.embed_dim, out_features=self.num_classes)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout()
        self.mouth_cls = nn.Linear(in_features=self.num_frames, out_features=self.num_classes)

    def forward(self, x, y, rois):
        x_t = self.timesformer.model.forward_features(x)

        fusion_1, y_out = self.fusion_1(x_t, y, rois)

        fusion_2, y_out = self.fusion_2(fusion_1, y_out, rois)

        fusion_3, y_out = self.fusion_3(fusion_2, y_out)

        out = self.fusion_4(fusion_3)
        out = self.trans_norm(out)

        cls = out[:, 0]
        cls_out = self.cls_head(cls)

        logits = self.dropout(self.avg_pool(y_out))

        # for feature visualization
        mouth_feature = y_out.permute(0, 3, 2, 1)
        mouth_feature = self.avg_pool(mouth_feature)
        mouth_feature = mouth_feature.squeeze(2).squeeze(2)

        logits = self.mouth_cls(logits.squeeze(2).squeeze(2))

        return cls, mouth_feature, cls_out, logits


class ConvTransBlock(nn.Module):

    def __init__(self, times_blocks, timesformer, end_points=None, i3d=None, num_frames=None, num_patches=None,
                 fusion_channel=256, down_ratio=None, fusion=False, up_stride=1, down_stride=1):
        super(ConvTransBlock, self).__init__()

        self.num_frames = num_frames
        self.num_patches = num_patches
        self.fusion_channel = fusion_channel
        self.timesformer = timesformer
        self.i3d = i3d
        self.times_blocks = times_blocks
        self.end_points = end_points
        self.down_ratio = down_ratio
        if self.down_ratio is not None:
            self.temporal = self.num_frames // self.down_ratio
        self.fusion = fusion
        self.up_stride = up_stride
        self.down_stride = down_stride
        self.fusion_block = Fusion_Block(in_channels=self.num_frames, num_frames=self.num_frames,
                                         num_patches=self.num_patches, fusion_channel=self.fusion_channel,
                                         down_ratio=self.down_ratio, fusion=self.fusion, up_stride=self.up_stride,
                                         down_stride=self.down_stride)

    def forward(self, x, y=None, rois=None):
        B, T, W = x.size(0), self.num_frames, self.num_patches

        # GAD-branch

        for i in self.times_blocks:
            frame_wise_cls_token, x = self.timesformer.model.blocks[i](x, B, T, W)

        if y is None:
            return x

        x_t = x[:, 1:]
        cls_token = x[:, 0].unsqueeze(1)
        x_t = rearrange(x_t, 'b (n t) m -> b t n m', b=B, t=T)

        # LRM-branch
        for end_point in self.end_points:
            y = self.i3d._modules[end_point](y)

        y_in = y

        if self.fusion is True:
            x_t = x_t

        # fusion block
        mouth2img_fusion, img2mouth_fusion = self.fusion_block(frame_wise_cls_token, x_t, y_in, rois)

        if self.fusion is True:
            x_t = rearrange(mouth2img_fusion, 'b c t h w -> b (h w t) c')
            x_t = torch.cat((cls_token, x_t), dim=1)
            mouth_fusion = img2mouth_fusion
        else:
            x_t = torch.cat((cls_token, mouth2img_fusion), dim=1)
            mouth_fusion = img2mouth_fusion

        return x_t, mouth_fusion


class Fusion_Block(nn.Module):
    def __init__(self, in_channels, num_frames, num_patches, fusion_channel=256, down_ratio=None, fusion=False,
                 up_stride=1, down_stride=1, dropout=0.5):
        super(Fusion_Block, self).__init__()

        self.in_channels = in_channels
        self.num_frames = num_frames
        self.num_patches = num_patches
        self.up_stride = 1
        self.embed_dim = 768
        self.fusion_channel = fusion_channel
        self.down_ratio = down_ratio
        if self.down_ratio is not None:
            self.temporal = self.num_frames // self.down_ratio
        self.fusion = fusion
        if self.fusion == True:
            self.fusion_channel = 1
        self.up_stride = up_stride
        self.down_stride = down_stride
        self.dropout = dropout

        self.channel_attention = ChannelAttention(self.in_channels)
        self.spatial_attention = SpatialAttention()

        self.expand_block = FCUUp(inplanes=self.embed_dim, outplanes=self.fusion_channel, up_stride=self.up_stride,
                                  down_ratio=self.down_ratio, num_patches=self.num_patches, num_frames=self.num_frames)
        self.squeeze_block = FCUDown(inplanes=self.fusion_channel, outplanes=self.fusion_channel,
                                     dw_stride=self.num_patches, fusion=self.fusion, down_stride=self.down_stride)

        self.img_cross_attention = ImageCrossModelAttention()
        self.mouth_cross_attention = MouthCrossModelAttention()

    def forward(self, cls_token, x, y, rois=None):
        B, T = x.size(0), self.num_frames

        # spatial-wise attention weights for LRM-branch
        y_in = rearrange(y, 'b c t h w -> (b t) c h w', b=B, t=y.size(2))
        y_attention = self.spatial_attention(y_in)
        y_attention = rearrange(y_attention, '(b t) c h w -> b c t h w', b=B)

        x_t = rearrange(x, 'b t (h w) n -> b n t h w', h=self.num_patches)

        if self.fusion is True:
            img2mouth_fusion = self.expand_block(cls_token, y)
            mouth2img_fusion = self.squeeze_block(y_attention, x_t, rois)

        else:
            y_out = self.squeeze_block(y, x)
            y_out = rearrange(y_out, 'b c t h w -> b t c h w')

            # cross attention for two branch
            mouth2img_fusion = self.img_cross_attention(x, y_out)
            img2mouth_fusion = self.mouth_cross_attention(y_out, x)

        return mouth2img_fusion, img2mouth_fusion


class FCUDown(nn.Module):
    """ CNN feature maps -> Transformer patch embeddings
    """

    def __init__(self, inplanes, outplanes, dw_stride, fusion=False, down_stride=None, act_layer=nn.GELU,
                 norm_layer=partial(nn.BatchNorm3d, eps=1e-6)):
        super(FCUDown, self).__init__()
        self.dw_stride = dw_stride
        self.fusion = fusion
        self.down_stride = down_stride
        if fusion == False:
            outplanes = 768
        self.conv_project = nn.Conv3d(inplanes, outplanes, kernel_size=1,
                                      stride=(1, self.down_stride, self.down_stride), padding=0)

        self.ln = norm_layer(outplanes)
        self.act = act_layer()

    def forward(self, y, x_t, rois=None):
        B = x_t.size(0)
        spatial_scale = 1 / 16
        y = self.act(self.ln(self.conv_project(y)))  # [N, C, T, H, W]
        y = F.interpolate(y, size=(16, y.size(-2), y.size(-1)))

        if self.fusion is True:
            full_face = x_t.clone()
            full_face = rearrange(full_face, 'b n t h w -> (b t) n h w')
            mouth_region = roi_align(full_face, rois, (7, 7), spatial_scale=spatial_scale, sampling_ratio=2)
            mouth_region = rearrange(mouth_region, '(b t) n h w -> b n t h w', b=B)

            # mouth weight * mouth region taken from full-face
            weighted_mouth = y * mouth_region
            # re-align mouth to the global representation
            new_face = torch.zeros(x_t.size()).to("cuda")
            for idx in range(len(rois)):
                roi = rois[idx]
                start_x = roi[0][0] * spatial_scale
                start_y = roi[0][1] * spatial_scale
                end_x = roi[0][2] * spatial_scale
                end_y = roi[0][3] * spatial_scale

                start_x_int = math.floor(start_x)
                end_x_int = math.ceil(end_x)
                start_y_int = math.floor(start_y)
                end_y_int = math.ceil(end_y)

                if start_x_int + 7 < 14:
                    mouth_l = start_x_int
                    mouth_r = start_x_int + 7
                else:
                    mouth_l = end_x_int - 7
                    mouth_r = end_x_int

                if start_y_int + 7 < 14:
                    mouth_t = start_y_int
                    mouth_b = start_y_int + 7
                else:
                    mouth_t = end_y_int - 7
                    mouth_b = end_y_int

                new_face[:, :, idx, mouth_t:mouth_b, mouth_l:mouth_r] = weighted_mouth[:, :, idx, :, :]
            x_out = new_face + x_t

        else:
            x_out = y

        return x_out


class FCUUp(nn.Module):
    """ Transformer patch embeddings -> CNN feature maps
    """

    def __init__(self, inplanes, outplanes, up_stride, down_ratio, num_patches, num_frames, act_layer=nn.ReLU,
                 norm_layer=partial(nn.BatchNorm3d, eps=1e-6), ):
        super(FCUUp, self).__init__()
        self.up_stride = up_stride
        self.down_ratio = down_ratio
        self.num_patches = num_patches
        self.num_frames = num_frames
        if self.down_ratio is not None:
            self.temporal = self.num_frames // self.down_ratio
        self.conv_project = nn.Conv3d(inplanes, outplanes, kernel_size=1, stride=(self.down_ratio, 1, 1), padding=0)
        self.bn = norm_layer(outplanes)
        self.act = act_layer()

    def forward(self, x_r, y):
        if self.up_stride == 0.5:
            return x_r
        B, _, C = x_r.shape
        x_r = x_r.permute(0, 2, 1).unsqueeze(3).unsqueeze(4)
        x_r = self.act(self.bn(self.conv_project(x_r)))

        x_fusion = F.interpolate(x_r, size=(
        x_r.size(2), self.num_patches * self.up_stride, self.num_patches * self.up_stride))

        return x_fusion + y


class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data, gain=0.02)

    def forward(self, x):
        avgout = self.fc(self.avg_pool(x))
        maxout = self.fc(self.max_pool(x))
        y = self.sigmoid(avgout + maxout)
        return x * y.expand_as(x)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data, gain=0.02)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class ImageCrossModelAttention(nn.Module):
    def __init__(self, inner_dim=256, dim=768, dropout=0.5, norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        super(ImageCrossModelAttention, self).__init__()
        self.to_qkv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.keyconv = nn.AdaptiveAvgPool3d((inner_dim, 1, 1))
        self.scale = inner_dim ** -0.5
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )
        self.ln = norm_layer(dim)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data, gain=0.02)

    def forward(self, x, y):
        kv = self.to_qkv(x).chunk(2, dim=-1)
        k, v = kv[0], kv[1]
        q = self.keyconv(y).squeeze(3).permute(0, 1, 3, 2)

        dots = einsum('b t n d, b t j d -> b t n j', q, k) * self.scale

        attn = dots.softmax(dim=-1)

        out = einsum('b t n j, b t j d -> b t n d', attn, v)
        out = self.ln(self.to_out(out))
        out = out + x
        out = rearrange(out, 'b t n d -> b (t n) d')

        return out


class MouthCrossModelAttention(nn.Module):
    def __init__(self, inner_dim=256, dim=768, dropout=0.5, norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        super(MouthCrossModelAttention, self).__init__()
        self.to_qkv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.keyconv = nn.AdaptiveAvgPool2d((1, inner_dim))
        self.scale = inner_dim ** -0.5
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )
        self.ln = norm_layer(dim)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data, gain=0.02)

    def forward(self, x, y):

        x = x.flatten(3).permute(0, 1, 3, 2)
        kv = self.to_qkv(x).chunk(2, dim=-1)
        k, v = kv[0], kv[1]
        q = self.keyconv(y).squeeze(3)

        dots = einsum('b t n d, b t j d -> b t n j', q, k) * self.scale
        attn = dots.softmax(dim=-1)

        out = einsum('b t n j, b t j d -> b t n d', attn, v)
        out = self.ln(self.to_out(out))
        out = out + x

        return out

