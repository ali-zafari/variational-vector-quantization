import torch
import torch.distributions as D
from torch.utils.data import DataLoader, Dataset
from lightning.pytorch import LightningDataModule

from banana import BananaDistribution


__all__ = [
    'Source',
    'get_torch_dist'
]


def get_torch_dist(dist, **kwargs):

    if dist == 'laplace':
        if kwargs:
            laplace = D.Laplace(loc=torch.tensor([kwargs.get('loc')]), scale=torch.tensor([kwargs.get('scale')]))
        else:
            laplace = D.Laplace(loc=torch.zeros([1]), scale=torch.ones([1]))
        return D.Independent(laplace, reinterpreted_batch_ndims=1)

    elif dist == 'banana':
        return BananaDistribution(**kwargs)

    elif dist == 'normal2d':
        return D.MultivariateNormal(**kwargs) if kwargs else D.MultivariateNormal(loc=torch.Tensor((0, 0)),
                                                                                  covariance_matrix=torch.eye(2))
    elif dist == 'normal':
        if kwargs:
            normal = D.Normal(loc=torch.tensor([kwargs.get('loc')]), scale=torch.tensor([kwargs.get('scale')]))
        else:
            normal = D.Normal(loc=torch.zeros([1]), scale=torch.ones([1]))
        return D.Independent(normal, reinterpreted_batch_ndims=1)
    else:
        raise Exception(f'Distribution {dist} is not valid.')


class SourceDataset(Dataset):

    def __init__(self, samples):
        super().__init__()
        self.samples = samples

    def __len__(self):
        return self.samples.size(0)

    def __getitem__(self, idx):
        return self.samples[idx]


class Source(LightningDataModule):
    def __init__(self, dist, dist_kwargs, num_workers,
                 num_train_data, train_batch_size,
                 num_valid_data, valid_batch_size,
                 ):
        super().__init__()

        self.dist = get_torch_dist(dist, **dist_kwargs)
        self.num_workers = num_workers

        self.num_train_data = num_train_data
        self.train_batch_size = train_batch_size
        self.num_valid_data = num_valid_data
        self.valid_batch_size = valid_batch_size

    def setup(self, stage):
        self.train_dataset = SourceDataset(self.dist.sample((self.num_train_data,)))
        self.valid_dataset = SourceDataset(self.dist.sample((self.num_valid_data,)))

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.train_batch_size,
                          shuffle=True,
                          num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset,
                          batch_size=self.valid_batch_size,
                          shuffle=False,
                          num_workers=self.num_workers)
