from argparse import ArgumentParser
from datetime import datetime
from typing import Optional
import torch
from datamodule import ZeroShotYahooDataModule
from lightningmodel import ZeroShotClassificationModel
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelSummary
from pytorch_lightning.callbacks import DeviceStatsMonitor
from pytorch_lightning.callbacks import StochasticWeightAveraging
from pytorch_lightning.cli import LightningCLI
import pdb
seed_everything(42)
# Set breakpoint
# pdb.set_trace()

# Viet tensorboard log
# tensorboard --logdir=lightning_logs/
def main():
    # cli = LightningCLI(UITQATransformer, UITQADataModule)

    parser = ArgumentParser()

    # add PROGRAM level args
    parser.add_argument("--conda_env", type=str, default="ligning")
    parser.add_argument('--model_checkpoint', type = str, default = 'facebook/bart-base')
    parser.add_argument('--debugging', type = bool, default = False)
    # add model specific args
    parser = ZeroShotClassificationModel.add_model_specific_args(parser)
    parser = ZeroShotYahooDataModule.add_model_specific_args(parser)
    # add all the available trainer options to argparse
    # ie: now --accelerator --devices --num_nodes ... --fast_dev_run all work in the cli
    parser = Trainer.add_argparse_args(parser,)

    args = parser.parse_args()
    print(args)
   
    dict_args = vars(args)

    dm = ZeroShotYahooDataModule(**dict_args)
    seen_types = dm.prepare_data()
    # dm.setup('fit')
    # example_batch = next(iter(dm.train_dataloader()))
    # print(example_batch)    
    model = ZeroShotClassificationModel(**dict_args, seen_types = seen_types)
    # print(model.training_step(example_batch,batch_idx=0))

    # early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=3, verbose=False, mode="min")

    trainer = Trainer.from_argparse_args(
        args, 
        default_root_dir = 'checkpoints/',
        accelerator="gpu",
        devices=[2],
        max_epochs = 10,
        precision=16,
        # fast_dev_run=True,
        # limit_train_batches=10, 
        # limit_val_batches=5,
        # profiler="simple",
        # callbacks = [
        #   early_stop_callback,
        #   DeviceStatsMonitor(),
        #   StochasticWeightAveraging(swa_lrs=1e-2)
        # ],
        # devices=1 if torch.cuda.is_available() else None,
        # auto_scale_batch_size = 'binsearch',
        # auto_lr_find=True,
    )

    # # Find batchsize or lr for training
    trainer.tune(model, datamodule=dm)

    # Run learning rate finder on notebook
    lr_finder = trainer.tuner.lr_find(model, datamodule=dm)
    # Results can be found in
    # print(lr_finder.suggestion())
    # Plot with
    fig = lr_finder.plot(suggest=True)
    fig.savefig('lr_find_suggest.png')
    # Pick point based on plot, or get suggestion
    new_lr = lr_finder.suggestion()

    # update hparams of the model
    model.hparams.lr = new_lr

    trainer.fit(model, datamodule=dm)

if __name__ == "__main__":
    main()