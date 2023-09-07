import dataclass
import model
import config
import pytorch_lightning as pl
from callbacks import MyCallbacks, EarlyStopping
import torch
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.profilers import PyTorchProfiler
import os
os.environ['KINETO_LOG_LEVEL'] = '3'
from pytorch_lightning.strategies.deepspeed import DeepSpeedStrategy


torch.set_float32_matmul_precision('high')  # To make lightning happy

if __name__ == "__main__":
    logger = TensorBoardLogger('tb_logs', name='minst_model_v1')

    profiler = PyTorchProfiler(
        on_trace_ready=torch.profiler.tensorboard_trace_handler("{}/profiler1".format(logger.log_dir)),
        schedule=torch.profiler.schedule(
            skip_first=10,
            wait=1,
            warmup=1,
            active=20),
    )

    strategy = DeepSpeedStrategy()

    model = model.NN(
        input_size=config.INPUT_SIZE,
        num_classes=config.NUM_CLASSES,
        lr=config.LEARNING_RATE,
    )

    datamodule = dataclass.MyMnistDataModule(
        data_dir=config.DATA_DIR,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
    )

    # When set 'ddp' set DEVICES = [0, 1, 2, ..] Number of GPU you want to use
    trainer = pl.Trainer(
        # strategy='ddp',     # DistributedDataParallel
        strategy=strategy,    # DeepSpeed DistributedDataParallel is more efficient
        profiler=profiler,
        logger=logger,
        min_epochs=1,
        max_epochs=config.EPOCHS,
        accelerator=config.ACCERLATOR,
        devices=config.DEVICES,
        precision=config.PRECISION,
        callbacks=[MyCallbacks(), EarlyStopping(monitor='val_loss')],
    )

    # Training model
    trainer.fit(model, datamodule=datamodule)

    # Evaluating model
    trainer.evaluating(model, datamodule=datamodule)

    # Testing model
    trainer.test(model, datamodule=datamodule)
