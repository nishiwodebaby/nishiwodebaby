import torch
import torch.nn as nn
import torch.nn.functional as F

class CSAM(nn.Module):
    """
    Collaborative Spatial-Channel Attention Module.
    Performs simultaneous channel and spatial attention.
    """
    def __init__(self, in_channels, reduction_ratio=16):
        super(CSAM, self).__init__()
        self.in_channels = in_channels
        self.reduction_ratio = reduction_ratio

        # Channel attention path
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

        # Spatial attention path
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x shape: [B, C, H, W]
        # Channel attention: [B, C, 1, 1]
        ca = self.channel_attention(x)

        # Spatial attention: [B, 1, H, W]
        sa = self.spatial_attention(x)

        # Apply both attentions and sum
        x_ca = x * ca
        x_sa = x * sa
        out = x_ca + x_sa
        return out

class EncoderBlock(nn.Module):
    """Encoder block with CSAM."""
    def __init__(self, in_channels, out_channels, use_csam=True):
        super(EncoderBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.use_csam = use_csam
        if use_csam:
            self.csam = CSAM(out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        if self.use_csam:
            x = self.csam(x)
        skip = x  # Save for skip connection
        x = self.pool(x)
        return x, skip

class DecoderBlock(nn.Module):
    """Decoder block with upsampling and skip connection."""
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip):
        x = self.up(x)
        # Handle potential size mismatch due to pooling/upsampling
        diff_h = skip.shape[2] - x.shape
        diff_w = skip.shape[3] - x.shape
        if diff_h > 0 or diff_w > 0:
            x = F.pad(x, [diff_w // 2, diff_w - diff_w // 2,
                          diff_h // 2, diff_h - diff_h // 2])
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x

class CSFNet(nn.Module):
    """
    Main CSF-Net architecture.
    """
    def __init__(self, config):
        super(CSFNet, self).__init__()
        model_cfg = config['model']
        self.encoder_channels = model_cfg['encoder_channels']
        self.decoder_channels = model_cfg['decoder_channels']
        self.use_csam = model_cfg['use_csam']
        self.num_classes = config['data']['num_classes']
        self.ffcl_embed_dim = model_cfg.get('ffcl_embed_dim', 128)

        # Input convolution (4 modalities -> first encoder channel)
        self.input_conv = nn.Conv2d(
            len(config['data']['modalities']),  # 4 modalities
            self.encoder_channels[0],
            kernel_size=3,
            padding=1
        )

        # Encoder
        self.encoder_blocks = nn.ModuleList()
        for i in range(len(self.encoder_channels) - 1):
            self.encoder_blocks.append(
                EncoderBlock(
                    self.encoder_channels[i],
                    self.encoder_channels[i + 1],
                    use_csam=self.use_csam
                )
            )

        # Bottleneck (no pooling after last encoder)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(self.encoder_channels[-1], self.encoder_channels[-1] * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.encoder_channels[-1] * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.encoder_channels[-1] * 2, self.encoder_channels[-1] * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.encoder_channels[-1] * 2),
            nn.ReLU(inplace=True)
        )

        # Projection head for FFCL (if used during training)
        self.ffcl_projection = nn.Sequential(
            nn.Linear(self.encoder_channels[-1] * 2, self.ffcl_embed_dim),
            nn.ReLU(),
            nn.Linear(self.ffcl_embed_dim, self.ffcl_embed_dim)
        )

        # Decoder
        self.decoder_blocks = nn.ModuleList()
        # First decoder takes bottleneck output and last skip
        self.decoder_blocks.append(
            DecoderBlock(self.encoder_channels[-1] * 2, self.decoder_channels[0](@ref)
        )
        # Remaining decoder blocks
        for i in range(1, len(self.decoder_channels)):
            self.decoder_blocks.append(
                DecoderBlock(self.decoder_channels[i-1], self.decoder_channels[i])
            )

        # Final convolution to produce segmentation map
        self.final_conv = nn.Conv2d(self.decoder_channels[-1], self.num_classes, kernel_size=1)

    def forward(self, x, return_features=False):
        """
        Args:
            x: Input tensor [B, 4, H, W] (4 modalities)
            return_features: If True, return bottleneck features for FFCL
        Returns:
            seg_logits: Segmentation logits [B, num_classes, H, W]
            bottleneck_feats: (Optional) Bottleneck features [B, C, H', W'] flattened
        """
        # Encoder path
        skips = []
        x = self.input_conv(x)
        for encoder_block in self.encoder_blocks:
            x, skip = encoder_block(x)
            skips.append(skip)

        # Bottleneck
        bottleneck = self.bottleneck(x)

        # Project for FFCL if needed
        bottleneck_feats = None
        if return_features:
            # Flatten spatial dimensions for contrastive learning
            b, c, h, w = bottleneck.shape
            projected = self.ffcl_projection(bottleneck.reshape(b, c, -1).transpose(1, 2))
            bottleneck_feats = projected.reshape(b, -1, self.ffcl_embed_dim)

        # Decoder path with skip connections
        x = bottleneck
        for i, decoder_block in enumerate(self.decoder_blocks):
            skip = skips[-(i+1)]  # Use skips in reverse order
            x = decoder_block(x, skip)

        # Final segmentation map
        seg_logits = self.final_conv(x)

        if return_features:
            return seg_logits, bottleneck_feats
        return seg_logits
