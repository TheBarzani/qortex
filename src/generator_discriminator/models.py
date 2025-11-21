import torch
import torch.nn as nn


class Generator(nn.Module):
    """
    Classical generator G(context, noise) -> next-day surface (scaled).

    - context: current surface (vector of size context_dim)
    - noise: latent vector (size latent_dim)
    - output: generated next-day surface (vector of size target_dim)
    """

    def __init__(
        self,
        context_dim: int,
        target_dim: int,
        latent_dim: int = 16,
        hidden_sizes=(128, 128),
    ):
        super().__init__()
        self.context_dim = context_dim
        self.target_dim = target_dim
        self.latent_dim = latent_dim

        layers = []
        in_dim = context_dim + latent_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h

        layers.append(nn.Linear(in_dim, target_dim))
        # No activation here; we train on scaled targets (roughly Gaussian)
        self.net = nn.Sequential(*layers)

    def forward(self, context: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """
        context: (batch_size, context_dim)
        noise:   (batch_size, latent_dim)

        returns: (batch_size, target_dim)
        """
        x = torch.cat([context, noise], dim=-1)
        return self.net(x)


class Discriminator(nn.Module):
    """
    Classical discriminator D(context, surface) -> logit (real vs fake).

    - context: current surface (vector of size context_dim)
    - surface: candidate next-day surface (vector of size target_dim)
    - output: scalar logit per sample (before sigmoid)
    """

    def __init__(
        self,
        context_dim: int,
        target_dim: int,
        hidden_sizes=(128, 128),
    ):
        super().__init__()
        self.context_dim = context_dim
        self.target_dim = target_dim

        layers = []
        in_dim = context_dim + target_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.LeakyReLU(0.2))
            in_dim = h

        layers.append(nn.Linear(in_dim, 1))  # logit
        self.net = nn.Sequential(*layers)

    def forward(self, context: torch.Tensor, surface: torch.Tensor) -> torch.Tensor:
        """
        context: (batch_size, context_dim)
        surface: (batch_size, target_dim)

        returns: (batch_size, 1) logits
        """
        x = torch.cat([context, surface], dim=-1)
        return self.net(x)