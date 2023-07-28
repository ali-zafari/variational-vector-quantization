import os
import abc

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import optim
from lightning.pytorch import LightningModule

__all__ = ['CompressionModel']


class CompressionModel(LightningModule, metaclass=abc.ABCMeta):

    def __init__(self, source, lmbda, distortion_fn):
        super().__init__()
        self.source = source
        self.lmbda = lmbda
        self.distortion_fn = distortion_fn
        self.save_hyperparameters(ignore=['distortion_fn', 'source'])

    @property
    def ndim_source(self):
        return self.source.event_shape[0]

    @abc.abstractmethod
    def quantize(self, x):
        raise NotImplementedError()

    @abc.abstractmethod
    def train_losses(self, x):
        raise NotImplementedError()

    @abc.abstractmethod
    def test_losses(self, x):
        raise NotImplementedError()

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1E-3)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'valid/loss'
            }
        }

    def training_step(self, batch, batch_idx):
        rates, distortions = self.train_losses(batch)
        losses = rates + self.lmbda * distortions
        loss = losses.mean()
        log_info = {
            "train/loss": loss,
            "train/rate": rates.mean(),
            "train/distortion": distortions.mean(),
        }
        self.log_dict(log_info)
        return loss

    def validation_step(self, batch, batch_idx):
        rates, distortions = self.test_losses(batch)
        losses = rates + self.lmbda * distortions
        log_info = {
            "valid/loss": losses.mean(),
            "valid/rate": rates.mean(),
            "valid/distortion": distortions.mean(),
        }
        self.log_dict(log_info)

    def _quantization_figure(self, intervals, figsize=(6, 6)):
        data = [torch.linspace(*i) for i in intervals]
        data = torch.meshgrid(*data, indexing='ij')
        data = torch.stack(data, dim=-1).to(self.device)
        codebook, rates, indexes = self.quantize(data)
        codebook = codebook.cpu().numpy()
        rates = rates.cpu().numpy()
        indexes = indexes.cpu().numpy()
        data_dist = torch.exp(self.source.log_prob(data.cpu())).cpu().numpy()
        data = data.cpu().numpy()
        counts = np.bincount(np.ravel(indexes), minlength=len(codebook))
        prior = 2 ** (-rates)

        if self.ndim_source == 1:
            boundaries = np.nonzero(indexes[1:] != indexes[:-1])[0]
            boundaries = (data[boundaries] + data[boundaries + 1]) / 2
            fig = plt.figure(figsize=figsize, dpi=200)
            plt.plot(data, data_dist, label="source")
            markers, stems, base = plt.stem(codebook[counts > 0], prior[counts > 0], label="codebook")
            plt.setp(markers, color="black")
            plt.setp(stems, color="black")
            plt.setp(base, linestyle="None")
            plt.xticks(np.sort(codebook[counts > 0].squeeze()))
            plt.grid(False, axis="x")
            for r in boundaries:
                plt.axvline(r, color="black", lw=1, ls=":", label="boundaries" if r == boundaries[0] else None)
            plt.xlim(np.min(data), np.max(data))
            plt.ylim(bottom=-.01)
            plt.legend(loc="upper left")
            plt.xlabel("source space")
            plt.tight_layout()
            epoch_num = self.current_epoch + 1 if self.global_step else 0
            plt.savefig(f'{self.logger.log_dir}/plots/quantization_plot_epoch_{epoch_num:03d}.png',
                        bbox_inches='tight')
            plt.title(f'Epoch {self.current_epoch:03d}')
            return fig

        elif self.ndim_source == 2:
            google_pink = (0xf4 / 255, 0x39 / 255, 0xa0 / 255)
            fig = plt.figure(figsize=figsize, dpi=200)
            vmax = data_dist.max()
            plt.imshow(
                data_dist, vmin=0, vmax=vmax, origin="lower",
                extent=(data[0, 0, 1], data[0, -1, 1], data[0, 0, 0], data[-1, 0, 0]))
            plt.contour(data[:, :, 1], data[:, :, 0], indexes, np.arange(len(codebook)) + .5,
                        colors=[google_pink], linewidths=.5)
            present_indices = counts > 0
            marker_sizes = -100 * np.log(1-prior[present_indices])
            plt.scatter(codebook[present_indices, 1], codebook[present_indices, 0], marker="o",
                        color=google_pink, s=marker_sizes)
            plt.axis("image")
            plt.grid(False)
            plt.xlim(data[0, 0, 1], data[0, -1, 1])
            plt.ylim(data[0, 0, 0], data[-1, 0, 0])
            plt.xlabel("source dimension 1")
            plt.ylabel("source dimension 2")
            plt.tight_layout()
            epoch_num = self.current_epoch + 1 if self.global_step else 0
            plt.savefig(f'{self.logger.log_dir}/plots/quantization_plot_epoch_{epoch_num:03d}.png',
                        bbox_inches='tight')
            plt.title(f'Epoch {self.current_epoch:03d}')
            return fig

        else:
            raise Exception('Quantization plot is only available for sources with dimension <= 2.')

    def plot_quantization(self, intervals):
        fig = self._quantization_figure(intervals)
        plt.show()
        plt.close(fig)

    def on_validation_start(self):
        os.makedirs(f'{self.logger.log_dir}/plots', exist_ok=True)
        intervals = self.ndim_source * [(-5, 7, 500)]
        fig = self._quantization_figure(intervals)
        self.logger.experiment.add_figure('quantize_plot', figure=fig, global_step=self.global_step)
        plt.close(fig)
