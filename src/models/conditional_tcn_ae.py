"""
SW-ADP Stage 4: Conditional TCN Autoencoder
Condition C: Encoder sees telemetry + SW (7 channels), decoder reconstructs telemetry only (1 channel).
"""
import torch
import torch.nn as nn


class CausalConv1d(nn.Module):
    """1D causal convolution with left-padding."""
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              dilation=dilation, padding=0)

    def forward(self, x):
        x = nn.functional.pad(x, (self.padding, 0))
        return self.conv(x)


class TCNBlock(nn.Module):
    """TCN residual block: 2x (CausalConv -> BN -> ReLU -> Dropout) + residual."""
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.1):
        super().__init__()
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        res = self.residual(x)
        out = self.dropout(self.relu(self.bn1(self.conv1(x))))
        out = self.dropout(self.relu(self.bn2(self.conv2(out))))
        return out + res


class ConditionalTCNAutoencoder(nn.Module):
    """
    Asymmetric TCN Autoencoder.

    Encoder input:  [batch, seq_len, input_dim]   (7: telem + 6 SW)
    Decoder output: [batch, seq_len, output_dim]   (1: telem only)

    Same architecture as TCNAutoencoder but with asymmetric input/output dims.
    The encoder processes all context; the decoder only targets telemetry.
    """
    def __init__(self, input_dim=7, output_dim=1, channels=[16, 32, 64],
                 kernel_size=3, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Encoder: input_dim -> channel stack
        enc_layers = []
        in_ch = input_dim
        for i, out_ch in enumerate(channels):
            dilation = 2 ** i
            enc_layers.append(TCNBlock(in_ch, out_ch, kernel_size, dilation, dropout))
            in_ch = out_ch
        self.encoder = nn.Sequential(*enc_layers)

        # Decoder: mirrored channels -> output_dim (NOT input_dim)
        dec_channels = list(reversed(channels))
        dec_layers = []
        for i, (in_ch, out_ch) in enumerate(zip(dec_channels[:-1], dec_channels[1:])):
            dilation = 2 ** (len(dec_channels) - 1 - i)
            dec_layers.append(TCNBlock(in_ch, out_ch, kernel_size, dilation, dropout))
        self.decoder = nn.Sequential(*dec_layers)
        self.output_proj = nn.Conv1d(dec_channels[-1], output_dim, 1)

    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, input_dim] (7 channels)
        Returns:
            reconstructed: [batch, seq_len, output_dim] (1 channel)
        """
        # Encode
        enc = self.encoder(x.transpose(1, 2))   # [batch, channels[-1], seq_len]
        # Decode
        dec = self.decoder(enc)                   # [batch, channels[0], seq_len]
        out = self.output_proj(dec)               # [batch, output_dim, seq_len]
        return out.transpose(1, 2)                # [batch, seq_len, output_dim]

    def get_model_size(self):
        n_params = sum(p.numel() for p in self.parameters())
        size_kb = n_params * 4 / 1024
        return n_params, size_kb
