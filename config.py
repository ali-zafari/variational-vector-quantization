class Config:
    seed = 1

    distribution = "normal2d"  # "laplace" / "banana" / "normal2d" / "normal"
    distribution_kwargs = {}

    model = "vecvq"
    lmbda = 10
    codebook_size = 7
    distortion = 'mse'

    num_train_data = 1_000_000
    num_valid_data = 100_000
    train_batch_size = 250
    valid_batch_size = 500
    num_workers = 8

    max_epochs = 100
    accelerator = 'gpu'
    devices = [0]
    log_cadence = 10
    validation_cadence = 1
