import os

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms

from ba3l.ingredients import datasets, models

from ba3l.experiment import Experiment
from ba3l.module import Ba3lModule
from sacred.config.utils import CMD
import pytorch_lightning as pl

from ba3l.trainer import Trainer as pzTrainer
from sacred.utils import apply_backspaces_and_linefeeds

ex = Experiment("Mnist_Ex1")


@ex.models.config
def def_conf():
    net = {"path": "cnn.Net", "c1": 1}


@ex.datasets.training.config
def training_cfg():
    batch_size = 3
    x = CMD("samplecmd")
    download = True
    num_workers = 12


@ex.datasets.test.config
def training_cfg():
    batch_size = 6
    download = True
    num_workers = 12


ex.datasets.training.dataset(
    MNIST, static_args=dict(transform=transforms.ToTensor()), root=os.getcwd()
)
ex.datasets.training.iter(DataLoader)

ex.datasets.test.dataset(
    MNIST,
    static_args=dict(transform=transforms.ToTensor()),
    root=os.getcwd(),
    train=False,
    test=True,
    validate=True,
)
ex.datasets.test.iter(DataLoader)


class M(Ba3lModule):
    def __init__(self, experiment):
        super(M, self).__init__(experiment)

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        # REQUIRED
        x, y = batch
        y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("train.loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        # OPTIONAL
        x, y = batch
        y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("validation.loss", loss, prog_bar=True, on_epoch=True, on_step=False)

        return {"val_loss": loss}

    def validation_end(self, outputs):
        # OPTIONAL
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        tensorboard_logs = {"val_loss": avg_loss}
        return {"avg_val_loss": avg_loss, "log": tensorboard_logs}

    def configure_optimizers(self):
        # REQUIRED
        # can return multiple optimizers and learning_rate schedulers
        # (LBFGS it is automatically supported, no need for closure function)
        return torch.optim.Adam(self.parameters(), lr=0.02)


#
# @ex.datasets.training.command
# def samplecmd(batch_size):
#     return batch_size * 100
#
#
# @ex.datasets.training.command
# def pr(x):
#     print("BSx100:", x)


@ex.command
def main(_run, _config, _log, _rnd, _seed):
    # force overriding the config, not logged = not recommended
    trainer = ex.get_trainer(max_epochs=1)

    trainer.fit(
        M(ex),
        train_dataloader=ex.get_train_dataloaders(),
        val_dataloaders=ex.get_val_dataloaders(),
    )

    return {"done": True}


@ex.command
def test_loaders():
    for i, b in enumerate(ex.datasets.training.get_iter()):
        print(b[1])
        break

    for i, b in enumerate(ex.datasets.test.get_iter()):
        print(b[1])
        break


@ex.automain
def default_command():
    return main()
