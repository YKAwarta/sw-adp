"""
SW-ADP Stage 4: TCN Autoencoder
Core model -- carries from Phase 1 PoC through Phase 2 ESP32 deployment.
1D dilated causal convolutions. Parallelizable, fixed memory footprint,
directly portable to TFLite-Micro.
"""
import torch
import torch.nn as nn


class CausalConv1d(nn.Module):
    """1D causal convolution with left-padding to preserve sequence length."""
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            dilation=dilation, padding=0  # we pad manually
        )

    def forward(self, x):
        # x: [batch, channels, seq_len]
        x = nn.functional.pad(x, (self.padding, 0))  # left-pad only (causal)
        return self.conv(x)


class TCNBlock(nn.Module):
    """
    Single TCN residual block:
    CausalConv -> BatchNorm -> ReLU -> Dropout -> CausalConv -> BatchNorm -> ReLU -> Dropout
    + residual connection (with 1x1 conv if channel dims differ)
    """
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.1):
        super().__init__()
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

        # Residual connection
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        # x: [batch, channels, seq_len]
        res = self.residual(x)
        out = self.dropout(self.relu(self.bn1(self.conv1(x))))
        out = self.dropout(self.relu(self.bn2(self.conv2(out))))
        return out + res


class TCNEncoder(nn.Module):
    def __init__(self, input_dim, channels=[16, 32, 64], kernel_size=3, dropout=0.1):
        super().__init__()
        layers = []
        in_ch = input_dim
        for i, out_ch in enumerate(channels):
            dilation = 2 ** i
            layers.append(TCNBlock(in_ch, out_ch, kernel_size, dilation, dropout))
            in_ch = out_ch
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        # x: [batch, seq_len, input_dim] -> transpose for Conv1d
        x = x.transpose(1, 2)  # [batch, input_dim, seq_len]
        return self.network(x)  # [batch, channels[-1], seq_len]


class TCNDecoder(nn.Module):
    """Mirror of encoder using transposed convolutions."""
    def __init__(self, output_dim, channels=[64, 32, 16], kernel_size=3, dropout=0.1):
        super().__init__()
        layers = []
        for i, (in_ch, out_ch) in enumerate(zip(channels[:-1], channels[1:])):
            dilation = 2 ** (len(channels) - 1 - i)
            layers.append(TCNBlock(in_ch, out_ch, kernel_size, dilation, dropout))

        self.network = nn.Sequential(*layers)
        self.output_proj = nn.Conv1d(channels[-1], output_dim, 1)

    def forward(self, x):
        # x: [batch, channels[0], seq_len]
        out = self.network(x)  # [batch, channels[-1], seq_len]
        out = self.output_proj(out)  # [batch, output_dim, seq_len]
        return out.transpose(1, 2)  # [batch, seq_len, output_dim]


class TCNAutoencoder(nn.Module):
    """
    TCN Autoencoder for time-series reconstruction.

    Architecture:
        Encoder: input -> TCNBlock(d=1) -> TCNBlock(d=2) -> TCNBlock(d=4)
        Bottleneck: encoded representation at each timestep
        Decoder: TCNBlock(d=4) -> TCNBlock(d=2) -> Linear projection -> output

    Receptive field with kernel_size=3, dilations [1,2,4]:
        Each block sees (kernel_size-1)*dilation*2 = 4*dilation timesteps
        Total receptive field = 1 + sum(2*(k-1)*d for each block) = 1 + 4+8+16 = 29 timesteps
        Sufficient for OPSSAT segments (median 70 timesteps).
    """
    def __init__(self, input_dim, channels=[16, 32, 64], kernel_size=3, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.encoder = TCNEncoder(input_dim, channels, kernel_size, dropout)
        self.decoder = TCNDecoder(input_dim, list(reversed(channels)), kernel_size, dropout)

    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, input_dim]
        Returns:
            reconstructed: [batch, seq_len, input_dim]
        """
        encoded = self.encoder(x)        # [batch, channels[-1], seq_len]
        reconstructed = self.decoder(encoded)  # [batch, seq_len, input_dim]
        return reconstructed

    def get_model_size(self):
        """Return model size in parameters and approximate KB."""
        n_params = sum(p.numel() for p in self.parameters())
        size_kb = n_params * 4 / 1024  # float32
        return n_params, size_kb
