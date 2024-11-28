from argparse import Namespace
from itertools import islice

import pandas as pd
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl

# from dataset_uno import ExternalDataset
from _dataset import ExternalDataset
from acsmodel import ACSModel

class rENEModel(pl.LightningModule):
    """
    rENE LightningModule class with relevant train/val/test steps and dataloaders.
    """

    def __init__(self, params: Namespace = None):
        """
        Parameters
        ----------
        params
            `Namespace` object containing the model hyperparameters.
            Should usually be generated automatically by `argparse`.
        """
        super(rENEModel, self).__init__()
        
        self.params = params
        self.last = None
        self.acsconv = False
        self.ene_weight = params.ene_weight
        
        try:
            self.v = params.verbose
        except:
            self.v = False
        
        self.acsconv = True
        self.model = ACSModel(pretrained=params.pretrained, act=params.activation, dropout=params.dropout, resnet=params.resnet, num_tasks=1)#.model_3d

        # LOSS function
        self.loss = F.mse_loss

    def forward(self, x):
        """
        Forward prop on samples
        """
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        y_ene, y_hat_ene = y[:,0], y_hat[:,0]

        loss = self.loss(y_hat_ene, y_ene) #* self.ene_weight + self.loss(y_hat_nm, y_nm)

        return {'loss': loss, 
                'y_hat_ene': y_hat_ene.detach(), 
                'y_ene': y_ene.detach()}

    def validation_step(self, batch, batch_idx):
        return self.training_step(batch, batch_idx)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        return {'y_hat': y_hat}
    
    @staticmethod
    def metrics(y_true, y_hat, stage=""):
        from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_curve, auc

        assert y_true.shape == y_hat.shape, f"y_true and y_hat do not match shapes: {y_true.shape} vs {y_hat.shape}"

        binarize = np.vectorize(lambda x: 1 if x > 0.5 else 0)
        acc = accuracy_score(y_true, binarize(y_hat))
        auroc = roc_auc_score(y_true, y_hat)
        pre, rec, _ = precision_recall_curve(y_true, y_hat)
        auprc = auc(rec, pre)
        
        if stage != "":
            stage += "_"
        
        return acc, auroc, auprc
    
    @staticmethod
    def outputs_to_metrics(outputs):
        loss        = torch.stack([j['loss'] for j in outputs]).mean()
        y_ene       = torch.cat([j['y_ene'] for j in outputs]).cpu()
        y_hat_ene   = torch.cat([j['y_hat_ene'] for j in outputs]).cpu()

        acc_ene, auroc_ene, auprc_ene   = self.metrics(y_ene, y_hat_ene)

        return loss, acc_ene, auroc_ene, auprc_ene

    @staticmethod
    def metrics_to_dict(loss, acc_ene, auroc_ene, auprc_ene, prefix='train'):
        return {f'{prefix}_loss': loss,
                f'{prefix}_auroc': auroc_ene, 
                f'{prefix}_auprc': auprc_ene, 
                f'{prefix}_acc': acc_ene}

    def training_epoch_end(self, outputs):
        metrics = self.outputs_to_metrics(outputs)
        self.log_dict(self.metrics_to_dict(*metrics, prefix='train'), prog_bar=True)
    
    def validation_epoch_end(self, outputs):
        metrics = self.outputs_to_metrics(outputs)
        self.log_dict(self.metrics_to_dict(*metrics, prefix='val'), prog_bar=True)
    
    def test_epoch_end(self, outputs):
        y_hat       = torch.cat([j['y_hat'] for j in outputs]).cpu()
        y_hat_ene = y_hat[:,0]
        
        patient_ids = self.test_dataset.image_list 
        if self.v: print(y_hat_ene.shape, len(patient_ids))
        pd.DataFrame.from_dict({'ID': patient_ids, 'ENE': y_hat_ene}).to_csv(self.params.pred_save_path, index=False) 

    def init_params(self, m: torch.nn.Module):
        """Initialize the parameters of a module.

        Parameters
        ----------
        m
            The module to initialize.

        Notes
        -----
        Convolutional layer weights are initialized from a normal distribution
        as described in [1]_ in `fan_in` mode. The final layer bias is
        initialized so that the expected predicted probability accounts for
        the class imbalance at initialization.

        References
        ----------
        .. [1] K. He et al. `Delving Deep into Rectifiers: Surpassing
           Human-Level Performance on ImageNet Classification`,
           arXiv:1502.01852 [cs], Feb. 2015.
        """

        if isinstance(m, nn.Conv3d):
            nn.init.kaiming_normal_(m.weight, a=.1)
        elif isinstance(m, nn.BatchNorm3d):
            nn.init.constant_(m.weight, 1.)
            nn.init.constant_(m.bias, 0.)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            nn.init.constant_(m.bias, 0.)

        self.first_params = torch.clone(next(islice(self.parameters(), 0, None))).to('cpu')

    def prepare_data(self):
        """Preprocess the data and create training, validation and test
        datasets.

        This method is called automatically by pytorch-lightning.
        """
        # print(self.para)
        # train_dataset = ExternalDataset(self.params.root_directory, 
        #                             input_size=self.params.input_size,
        #                             dataaug=self.params.dataaug,
        #                             num_workers=self.params.num_workers,
        #                             acsconv=self.acsconv,
        #                             split='train')
        
        test_dataset  = ExternalDataset(self.params.test_root, 
                                        self.params.test_mask, 
                                        testaug=self.params.testaug)    
        
        # self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        # print("\n\n\n\n", self.model.model_3d.state_dict(), "SDDD\n\n")

        print('Test data size:', len(test_dataset))
        
    
    def test_dataloader(self):
        """This method is called automatically by pytorch-lightning."""
        return DataLoader(self.test_dataset,
                              batch_size=self.params.batch_size,
                              num_workers=self.params.num_workers,
                              shuffle=False)
    
    def configure_optimizers(self):
        """This method is called automatically by pytorch-lightning."""
        optimizer = torch.optim.Adam(self.parameters(), 
                                         lr=self.params.lr,
                                         weight_decay=self.params.weight_decay)