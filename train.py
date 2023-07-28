import torch
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping

from config import Config
from source import Source, get_torch_dist
from models import VECVQ


def sse(x, x_hat):
    return torch.sum((x-x_hat)**2, dim=-1)


def mse(x, x_hat):
    return torch.mean((x-x_hat)**2, dim=-1)


def mae(x, x_hat):
    return torch.mean(torch.abs(x-x_hat), dim=-1)


available_distortions = {
    'sse': sse,
    'mse': mse,
    'mae': mae
}


def get_desired_model(config):
    if config.model == "vecvq":
        return VECVQ(codebook_size=Config.codebook_size,
                     source=get_torch_dist(Config.distribution, **Config.distribution_kwargs),
                     lmbda=Config.lmbda,
                     distortion_fn=available_distortions[Config.distortion])
    else:
        raise ModuleNotFoundError(f'Requested model {config.model} not found.')


def main():

    if Config.seed is not None:
        seed_everything(Config.seed)

    model_name = f'{Config.model}-{Config.distribution}-lambda={Config.lmbda}'
    dir_name = 'ckpt' + '/' + model_name

    data = Source(dist=Config.distribution,
                  dist_kwargs=Config.distribution_kwargs,
                  num_workers=Config.num_workers,
                  num_train_data=Config.num_train_data, train_batch_size=Config.train_batch_size,
                  num_valid_data=Config.num_valid_data, valid_batch_size=Config.valid_batch_size)
    model = get_desired_model(Config)

    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        monitor='valid/loss',
        mode='min',
        filename='epoch={epoch}-val_loss={valid/loss:.4f}-best',
        auto_insert_metric_name=False,
        save_on_train_epoch_end=True,
        save_last=True,
        verbose=True
    )
    checkpoint_callback.CHECKPOINT_NAME_LAST = 'epoch={epoch}-loss={train/loss:.4f}-last'
    early_stop_callback = EarlyStopping(monitor='valid/loss', min_delta=1E-6, patience=10, verbose=True)

    trainer = Trainer(
        default_root_dir=dir_name,
        accelerator=Config.accelerator,
        devices=Config.devices,
        max_epochs=Config.max_epochs,
        callbacks=[LearningRateMonitor(logging_interval='epoch'), early_stop_callback, checkpoint_callback],
        check_val_every_n_epoch=Config.validation_cadence,
        log_every_n_steps=Config.log_cadence,
        num_sanity_val_steps=-1
    )
    trainer.logger._default_hp_metric = None
    trainer.fit(model, data)


if __name__ == "__main__":
    main()
