import torch
from torch import nn, randn

from .compression_model import CompressionModel

__all__ = ['VECVQ']


class VECVQ(CompressionModel):

    def __init__(self, codebook_size, **kwargs):
        super().__init__(**kwargs)
        self.codebook_size = int(codebook_size)
        code_book_init = self.source.sample((self.codebook_size,))
        logits_init = randn(size=(self.codebook_size,))
        self.codebook = nn.Parameter(code_book_init)
        self._logits = nn.Parameter(logits_init)

    def all_rd(self, x):
        rates = - nn.functional.log_softmax(self._logits, dim=0) / torch.log(torch.tensor(2))  # SUM(-log2 prob)
        distortions = self.distortion_fn(x.unsqueeze(dim=-2), self.codebook)
        return rates, distortions

    @torch.no_grad()
    def quantize(self, x):
        rates, distortions = self.all_rd(x)
        all_rd = rates + self.lmbda * distortions
        indexes = torch.argmin(all_rd, dim=-1)
        return self.codebook, rates, indexes

    def test_losses(self, x):
        rates, distortions = self.all_rd(x)
        all_rd = rates + self.lmbda * distortions
        indexes = torch.argmin(all_rd, dim=-1)
        rates_indexed = rates[indexes]
        distortions_indexed = torch.gather(distortions, dim=-1, index=indexes.unsqueeze(dim=1)).squeeze()
        return rates_indexed, distortions_indexed

    train_losses = test_losses
