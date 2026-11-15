import torch
import torch.nn as nn
import torch.nn.functional as F

class SparseAutoencoder(nn.Module):
    def __init__(self, d_in=384, expansion_factor=16, l1_coeff=3e-4):
        super().__init__()
        self.d_in = d_in
        self.d_hidden = d_in * expansion_factor  # e.g., 384 * 16 = 6144 features
        self.l1_coeff = l1_coeff

        # Encoder and Decoder weights
        self.W_enc = nn.Parameter(torch.randn(d_in, self.d_hidden) * (1.0 / d_in**0.5))
        self.b_enc = nn.Parameter(torch.zeros(self.d_hidden))
        self.W_dec = nn.Parameter(self.W_enc.data.T.clone())
        self.b_dec = nn.Parameter(torch.zeros(d_in))

        # Normalize decoder columns to unit norm
        self.normalize_decoder()

    @torch.no_grad()
    def normalize_decoder(self):
        norms = self.W_dec.norm(dim=1, keepdim=True)
        self.W_dec.div_(norms.clamp(min=1e-8))

    @torch.no_grad()
    def remove_parallel_decoder_grad(self):
        W = self.W_dec
        grad = W.grad

        if grad is None:
            return

        parallel = (grad * W).sum(dim=1, keepdim=True)
        grad.sub_(parallel * W)

    def forward(self, x):
        # Center input using decoder bias (standard Anthropic/OpenAI SAE convention)
        x_cent = x - self.b_dec
        # If tuning l1_coeff causes dead neurons, replace F.relu(...) with a TopK activation (e.g., keeping only the top k = 32 activations per token).

        z = F.relu(x_cent @ self.W_enc + self.b_enc)
        x_hat = z @ self.W_dec + self.b_dec

        # Reconstruction loss (MSE) + Sparsity loss (L1)
        mse_loss = F.mse_loss(x_hat, x)
        l1_loss = self.l1_coeff * z.abs().sum(dim=-1).mean()
        loss = mse_loss + l1_loss
        return x_hat, z, loss, mse_loss, l1_loss
