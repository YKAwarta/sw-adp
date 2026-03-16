"""
SW-ADP Stage 4: Conditional LSTM Autoencoder
Condition C: Encoder sees telemetry + SW (7 channels), decoder reconstructs telemetry only (1 channel).
SW context informs the model's expectation of normal telemetry without diluting the anomaly score.
"""
import torch
import torch.nn as nn


class ConditionalLSTMAutoencoder(nn.Module):
    """
    Asymmetric LSTM Autoencoder.
    
    Encoder input:  [batch, seq_len, input_dim]   (7: telem + 6 SW)
    Decoder output: [batch, seq_len, output_dim]   (1: telem only)
    
    The encoder sees the full context. The decoder only reconstructs telemetry.
    Reconstruction error is computed on telemetry alone, so SW features inform
    the latent representation without diluting the anomaly signal.
    """
    def __init__(self, input_dim=7, output_dim=1, hidden_dim=64, n_layers=2, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        # Encoder sees ALL features (telem + SW)
        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )

        # Decoder reconstructs ONLY telemetry
        self.decoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.output_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, input_dim] (all 7 channels)
        Returns:
            reconstructed: [batch, seq_len, output_dim] (telemetry only)
        """
        batch_size, seq_len, _ = x.shape

        # Encode full context
        _, (h_n, c_n) = self.encoder(x)

        # Decode: repeat bottleneck across timesteps
        bottleneck = h_n[-1].unsqueeze(1).repeat(1, seq_len, 1)
        dec_out, _ = self.decoder(bottleneck, (h_n, c_n))
        reconstructed = self.output_proj(dec_out)  # [batch, seq_len, 1]

        return reconstructed

    def get_model_size(self):
        n_params = sum(p.numel() for p in self.parameters())
        size_kb = n_params * 4 / 1024
        return n_params, size_kb
