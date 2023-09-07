import dataclass
import model
import config
import pytorch_lightning as pl


if __name__ == "__main__":
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
    trainer = pl.Trainer(
        min_epochs=1,
        max_epochs=config.EPOCHS,
        accelerator=config.ACCERLATOR,
        devices=config.DEVICES,
        precision=config.PRECISION,
    )

    # Training model
    trainer.fit(model, datamodule=datamodule)
