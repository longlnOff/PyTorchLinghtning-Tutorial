from pytorch_lightning.callbacks import EarlyStopping, Callback
import pytorch_lightning as pl


class MyCallbacks(Callback):
    def __init__(self) -> None:
        super().__init__()

    def on_train_start(self, trainer, pl_module) -> None:
        print('Start training!')


    def on_train_end(self, trainer, pl_module) -> None:
        print('End training!')