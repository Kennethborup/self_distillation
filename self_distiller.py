import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.models import resnet50, resnet34, resnet18
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from pytorch_lightning import metrics, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from datetime import datetime
import argparse

class OneHot(object):
    def __init__(self, classes):
        self.classes = classes

    def __call__(self, y):
        return torch.zeros((1, self.classes)).scatter_(1, torch.tensor([y]).view(1,-1), 1).squeeze(0)

class LitResnet(pl.LightningModule):
    def __init__(self, hparams, teacher=None):
        super().__init__()
        # Save hyperparameters
        self.save_hyperparameters(hparams)

        self.classes = 10
        if self.hparams.resnet == 50:
            resnet = resnet50(pretrained=self.hparams.pretrained)
            resnet.fc = nn.Linear(in_features=2048, out_features=self.classes, bias=True)
        elif self.hparams.resnet == 34:
            resnet = resnet34(pretrained=self.hparams.pretrained)
            resnet.fc = nn.Linear(in_features=512, out_features=self.classes, bias=True)
        elif self.hparams.resnet == 18:
            resnet = resnet18(pretrained=self.hparams.pretrained)
            resnet.fc = nn.Linear(in_features=512, out_features=self.classes, bias=True)
        else:
            raise ValueError('Not valid resnet model')
        self.model = resnet

        self.distill = True if teacher is not None else False
        self.teacher = teacher

        # Metrics
        self.train_acc = metrics.Accuracy()
        self.val_acc = metrics.Accuracy()
        self.test_acc = metrics.Accuracy()

        # For training loss monitoring
        self.loss = 1e10

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--alpha', default=1.0, type=float, help='weight for distillation loss')
        parser.add_argument('--reg_coef', default=1e-4, type=float, help='weight for l2 regularization')
        parser.add_argument('--resnet', default=50, type=int, choices=[18, 34, 50], help='type of resnet to use')
        parser.add_argument('--lr', default=1e-3, type=float)
        parser.add_argument('--pretrained', default=False, dest='pretrained', action='store_true')
        return parser

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        preds = self.model(x)
        return preds

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        data, target = batch
        pred = self.model(data)
        loss = F.mse_loss(F.softmax(pred, dim=1), target)

        if self.distill:
            loss *= self.hparams.alpha
            teacher_target = self.teacher(data)
            distill_loss = F.mse_loss(F.softmax(pred, dim=1), F.softmax(teacher_target, dim=1))
            loss += (1-self.hparams.alpha)*distill_loss

        loss += self.hparams.reg_coef * sum([torch.norm(param) for param in model.model.parameters() if param.requires_grad])

        # Logging to TensorBoard by default
        self.loss = loss
        self.log('train_loss', loss)

        return {'loss' : loss, 'pred' : pred, 'target' : target}

    def validation_step(self, batch, batch_idx):
        data, target = batch
        pred = self.model(data)
        loss = F.mse_loss(F.softmax(pred, dim=1), target)

        # Logging to TensorBoard by default
        self.log('val_loss', loss)

        return {'loss' : loss, 'pred' : pred, 'target' : target}

    def test_step(self, batch, batch_idx):
        data, target = batch
        pred = self.model(data)
        loss = F.mse_loss(F.softmax(pred, dim=1), target)

        # Logging to TensorBoard by default
        self.log('test_loss', loss)

        return {'loss' : loss, 'pred' : pred, 'target' : target}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparams.lr)
        return optimizer

    def training_step_end(self, outputs):
        #update and log
        self.train_acc(outputs['pred'], torch.argmax(outputs['target'], dim=1))
        self.log('train_acc', self.train_acc, on_step=True, on_epoch=True)
        return outputs['loss']

    def validation_step_end(self, outputs):
        #update and log
        self.val_acc(outputs['pred'], torch.argmax(outputs['target'], dim=1))
        self.log('val_acc', self.val_acc)
        return outputs['loss']

    def test_step_end(self, outputs):
        #update and log
        self.test_acc(outputs['pred'], torch.argmax(outputs['target'], dim=1))
        self.log('test_acc', self.test_acc)
        return outputs['loss']


class CIFAR10DataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, batch_size, num_workers=4):
        super().__init__()
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.workers = num_workers

        self.train_preprocess = transforms.Compose([
            transforms.RandomCrop(32, padding=4), # pad all sides with 4 zeroes for 40x40 image size and random crop to 32x32 image
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

        self.test_preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    def prepare_data(self):
        CIFAR10(self.data_dir, train=True, download=True)
        CIFAR10(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            cifar_full = CIFAR10(self.data_dir, train=True, transform=self.train_preprocess, target_transform=OneHot(10))
            self.cifar_train, self.cifar_val = random_split(cifar_full, [50000, 0])
            self.dims = tuple(self.cifar_train[0][0].shape)

            self.cifar_test = CIFAR10(self.data_dir, train=False, transform=self.test_preprocess, target_transform=OneHot(10))

        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.cifar_test = CIFAR10(self.data_dir, train=False, transform=self.test_preprocess, target_transform=OneHot(10))
            self.dims = tuple(self.cifar_test[0][0].shape)

    def train_dataloader(self):
        return DataLoader(self.cifar_train, batch_size=self.batch_size, num_workers=self.workers)

    def val_dataloader(self):
#        return DataLoader(self.cifar_val, batch_size=self.batch_size, num_workers=self.workers)
        return DataLoader(self.cifar_test, batch_size=self.batch_size, num_workers=self.workers)

    def test_dataloader(self):
        return DataLoader(self.cifar_test, batch_size=self.batch_size, num_workers=self.workers)


# Parse arguments
parser = argparse.ArgumentParser(prog='Mobahi', description='Replication of Mobahi')

# add PROGRAM level args
parser.add_argument('--seed', type=int, default=None)
parser.add_argument('--batch-size', type=int, default=16)
parser.add_argument('--num-workers', type=int, default=4)
parser.add_argument('--distillation-steps', type=int, default=5)
parser.add_argument('--teacher-ckpt', type=str, default=None)
parser.add_argument('--teacher-step', type=int, default=0, help='Last (distillation) step the teacher was trained at (starts at 0)')


# add model specific args
parser = LitResnet.add_model_specific_args(parser)

# add all the available trainer options to argparse. ie: now --gpus --num_nodes ... --fast_dev_run all work in the cli
parser = pl.Trainer.add_argparse_args(parser)

args = parser.parse_args()
args.time = datetime.now().strftime('%d%m%Y_%H%M%S')
args.seed = torch.randint(0, int(1e4), (1,)).item() if args.seed is None else args.seed

seed_everything(args.seed)

# Load teacher if provided
if args.teacher_ckpt is not None:
    teacher = LitResnet.load_from_checkpoint(args.teacher_ckpt, strict=False) # strict allow us to ignore teacher weights in the teacher state_dict
    teacher.freeze()
    distillation_start_step = args.teacher_step + 1 # distillation step is the one following the teacher step
    args.time = teacher.hparams.time # Put new steps in same log folder as the teacher
else:
    teacher = None
    distillation_start_step = 0

cifar10 = CIFAR10DataModule('data/', args.batch_size, args.num_workers)
for step in range(distillation_start_step, distillation_start_step+args.distillation_steps):
    # Prepare logging
    logger = TensorBoardLogger("logs", name=args.time+f'_{args.alpha}_{args.lr}_{args.seed}', version=f'step_{step}')
    checkpoint_top_callback = ModelCheckpoint(
        monitor='val_acc',
        dirpath=f'checkpoints/{args.time}_{args.alpha}_{args.lr}_{args.seed}_{step}',
        filename='{epoch:02d}-{val_acc:.2f}',
        save_last=True,
        save_top_k=2,
        mode='max',
    )

    # Prepare model
    model = LitResnet(args, teacher=teacher)
    trainer = pl.Trainer.from_argparse_args(args,
                                            logger=logger,
                                            callbacks=[checkpoint_top_callback])

    # Train model
    trainer.fit(model, cifar10)

    # Update teacher for next step
    teacher = model # Teacher for next step
    teacher.freeze() # Freeze teacher
    teacher.teacher = None # Remove previous teacher from memory
